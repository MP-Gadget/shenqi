#include "localtreewalk2.h"
#include "omp.h"
#include "utils/endrun.h"
#include "utils/mymalloc.h"

/*!< Memory factor to leave for (N imported particles) > (N exported particles). */
static double ImportBufferBoost;
/* 7/9/24: The code segfaults if the send/recv buffer is larger than 4GB in size.
 * Likely a 32-bit variable is overflowing but it is hard to debug. Easier to enforce a maximum buffer size.*/
static size_t MaxExportBufferBytes = 3584*1024*1024L;

#define FACT1 0.366025403785    /* FACT1 = 0.5 * (sqrt(3)-1) */

#ifdef DEBUG
/*
 * for debugging
 */
#define WATCH { \
        printf("WorkSet[0] = %d (%d) %s:%d\n", WorkSet ? WorkSet[0] : 0, WorkSetSize, __FILE__, __LINE__); \
    }
static TreeWalk * GDB_current_ev = NULL;
#endif


/*Initialise global treewalk parameters*/
void set_treewalk_params(ParameterSet * ps)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask == 0)
        ImportBufferBoost = param_get_double(ps, "ImportBufferBoost");
    MPI_Bcast(&ImportBufferBoost, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

/* This function is to allow a test which fills up the exchange buffer*/
void treewalk_set_max_export_buffer(const size_t maxbuf)
{
    MaxExportBufferBytes = maxbuf;
}

size_t compute_bunchsize(const size_t query_type_elsize, const size_t result_type_elsize, const char * const ev_label)
{
    /*The amount of memory eventually allocated per tree buffer*/
    size_t bytesperbuffer = sizeof(struct data_index) + query_type_elsize + result_type_elsize;
    /*This memory scales like the number of imports. In principle this could be much larger than Nexport
     * if the tree is very imbalanced and many processors all need to export to this one. In practice I have
     * not seen this happen, but provide a parameter to boost the memory for Nimport just in case.*/
    bytesperbuffer += ceil(ImportBufferBoost * (query_type_elsize + result_type_elsize));
    /*Use all free bytes for the tree buffer, as in exchange. Leave some free memory for array overhead.*/
    size_t freebytes = (size_t) mymalloc_freebytes();
    freebytes -= 4096 * 10 * bytesperbuffer;

    size_t BunchSize = (size_t) floor(((double)freebytes)/ bytesperbuffer);
    if(BunchSize * query_type_elsize > MaxExportBufferBytes)
        BunchSize = MaxExportBufferBytes / query_type_elsize;
    /* Per thread*/
    BunchSize /= omp_get_max_threads();

    if(freebytes <= 4096 * bytesperbuffer || BunchSize < 100) {
        endrun(1231245, "Not enough free memory in %s to export particles: needed %ld bytes have %ld. can export %ld \n", ev_label, bytesperbuffer, freebytes, BunchSize);
    }
    return BunchSize;
}

template <typename NgbIterType,typename QueryType,typename ResultType>
LocalTreeWalk<NgbIterType, QueryType, ResultType>::LocalTreeWalk(const int i_mode, const ForceTree * const i_tree, const char * const i_ev_label, int * Ngblist, data_index ** ExportTable_thread):
 mode(i_mode), maxNinteractions(0), minNinteractions(1L<<45), Ninteractions(0), Nexport(0), tree(i_tree), ev_label(i_ev_label),
 BunchSize(compute_bunchsize(sizeof(QueryType), sizeof(ResultType), i_ev_label))
{
    const size_t thread_id = omp_get_thread_num();
    NThisParticleExport = 0;
    nodelistindex = 0;
    DataIndexTable = NULL;
    if(ExportTable_thread)
        DataIndexTable = ExportTable_thread[thread_id];
    ngblist = NULL;
    if(Ngblist)
        ngblist = Ngblist + thread_id * tree->NumParticles;
}

/* export a particle at target and no, thread safely
 *
 * This can also be called from a nonthreaded code
 *
 * */
template <typename NgbIterType, typename QueryType, typename ResultType>
int LocalTreeWalk<NgbIterType, QueryType, ResultType>::export_particle(const int no)
{
    if(mode != TREEWALK_TOPTREE || no < tree->lastnode) {
        endrun(1, "Called export not from a toptree.\n");
    }
    if(!DataIndexTable)
        endrun(1, "DataIndexTable not allocated, treewalk_export_particle called in the wrong way\n");
    if(no - tree->lastnode > tree->NTopLeaves)
        endrun(1, "Bad export leaf: no = %d lastnode %ld ntop %d target %d\n", no, tree->lastnode, tree->NTopLeaves, target);
    const int task = tree->TopLeaves[no - tree->lastnode].Task;
    /* This index is a unique entry in the global DataIndexTable.*/
    size_t nexp = Nexport;
    /* If the last export was to this task, we can perhaps just add this export to the existing NodeList. We can
     * be sure that all exports of this particle are contiguous.*/
    if(NThisParticleExport >= 1 && DataIndexTable[nexp-1].Task == task) {
#ifdef DEBUG
        /* This is just to be safe: only happens if our indices are off.*/
        if(DataIndexTable[nexp - 1].Index != target)
            endrun(1, "Previous of %ld exports is target %d not current %d\n", NThisParticleExport, DataIndexTable[nexp-1].Index, target);
#endif
        if(nodelistindex < NODELISTLENGTH) {
#ifdef DEBUG
            if(DataIndexTable[nexp-1].NodeList[nodelistindex] != -1)
                endrun(1, "Current nodelist %ld entry (%d) not empty!\n", nodelistindex, DataIndexTable[nexp-1].NodeList[nodelistindex]);
#endif
            DataIndexTable[nexp-1].NodeList[nodelistindex] = tree->TopLeaves[no - tree->lastnode].treenode;
            nodelistindex++;
            return 0;
        }
    }
    /* out of buffer space. Need to interrupt. */
    if(Nexport >= BunchSize) {
        return -1;
    }
    DataIndexTable[nexp].Task = task;
    DataIndexTable[nexp].Index = target;
    DataIndexTable[nexp].NodeList[0] = tree->TopLeaves[no - tree->lastnode].treenode;
    int i;
    for(i = 1; i < NODELISTLENGTH; i++)
        DataIndexTable[nexp].NodeList[i] = -1;
    Nexport++;
    nodelistindex = 1;
    NThisParticleExport++;
    return 0;
}

/* Do the regular particle visit with some extra cleanup of the particle export table for the toptree walk */
template <typename NgbIterType, typename QueryType, typename ResultType>
int LocalTreeWalk<NgbIterType, QueryType, ResultType>::toptree_visit(const QueryType& input, ResultType * output)
{
    /* Reset the number of exported particles.*/
    NThisParticleExport = 0;
    const int rt = visit(input, output);
    if(NThisParticleExport > 1000)
        message(5, "%ld exports for particle %d! Odd.\n", NThisParticleExport, target);
    /* If we filled up, we need to remove the partially evaluated last particle from the export list,
    * save the partially evaluated chunk, and leave this loop.*/
    if(rt < 0) {
        //message(5, "Export buffer full for particle %d chnk: %ld -> %ld on thread %d with %ld exports\n", i, chnk, end, tid, lv->NThisParticleExport);
        /* Drop partial exports on the current particle, whose toptree will be re-evaluated*/
        Nexport -= NThisParticleExport;
        /* Check that the final export in the list is indeed from a different particle*/
        if(NThisParticleExport > 0 && DataIndexTable[Nexport-1].Index >= target)
            endrun(5, "Something screwed up in export queue: nexp %ld (local %ld) last %d < index %d\n", Nexport,
                NThisParticleExport, target, DataIndexTable[Nexport-1].Index);
        /* Check that the earliest dropped export in the list is from the same particle*/
        if(NThisParticleExport > 0 && DataIndexTable[Nexport].Index != target)
            endrun(5, "Something screwed up in export queue: nexp %ld (local %ld) last %d != index %d\n", Nexport,
                NThisParticleExport, target, DataIndexTable[Nexport].Index);
    }
    return rt;
}

/**
 * Cull a node.
 *
 * Returns 1 if the node shall be opened;
 * Returns 0 if the node has no business with this query.
 */
static int
cull_node(const double * const Pos, const double BoxSize, const double Hsml, const NgbTreeFindSymmetric symmetric, const struct NODE * const current)
{
    double dist;
    if(symmetric == NGB_TREEFIND_SYMMETRIC) {
        dist = DMAX(current->mom.hmax, Hsml) + 0.5 * current->len;
    } else {
        dist = Hsml + 0.5 * current->len;
    }

    double r2 = 0;
    double dx = 0;
    /* do each direction */
    int d;
    for(d = 0; d < 3; d ++) {
        dx = NEAREST(current->center[d] - Pos[d], BoxSize);
        if(dx > dist) return 0;
        if(dx < -dist) return 0;
        r2 += dx * dx;
    }
    /* now test against the minimal sphere enclosing everything */
    dist += FACT1 * current->len;

    if(r2 > dist * dist) {
        return 0;
    }
    return 1;
}

/*****
 * This is the internal code that looks for particles in the ngb tree from
 * searchcenter upto hsml. if iter->symmetric is NGB_TREE_FIND_SYMMETRIC, then upto
 * max(Part[other].Hsml, iter->Hsml).
 *
 * Particle that intersects with other domains are marked for export.
 * The hosting nodes (leaves of the global tree) are exported as well.
 *
 * For all 'other' particle within the neighbourhood and are local on this processor,
 * this function calls the ngbiter member of the TreeWalk object.
 * iter->base.other, iter->base.dist iter->base.r2, iter->base.r, are properly initialized.
 *
 * */
 template <typename NgbIterType, typename QueryType, typename ResultType>
 int LocalTreeWalk<NgbIterType, QueryType, ResultType>::ngb_treefind_threads(const QueryType& I,
        NgbIterType * iter,
        int startnode)
{
    int no;
    int numcand = 0;

    const double BoxSize = tree->BoxSize;

    no = startnode;

    while(no >= 0)
    {
        if(node_is_particle(no, tree)) {
            int fat = force_get_father(no, tree);
            endrun(12312, "Particles should be added before getting here! no = %d, father = %d (ptype = %d) start=%d mode = %d\n", no, fat, tree->Nodes[fat].f.ChildType, startnode, mode);
        }
        if(node_is_pseudo_particle(no, tree)) {
            int fat = force_get_father(no, tree);
            endrun(12312, "Pseudo-Particles should be added before getting here! no = %d, father = %d (ptype = %d)\n", no, fat, tree->Nodes[fat].f.ChildType);
        }

        struct NODE *current = &tree->Nodes[no];

        /* When walking exported particles we start from the encompassing top-level node,
         * so if we get back to a top-level node again we are done.*/
        if(mode == TREEWALK_GHOSTS) {
            /* The first node is always top-level*/
            if(current->f.TopLevel && no != startnode) {
                /* we reached a top-level node again, which means that we are done with the branch */
                break;
            }
        }

        if(0 == cull_node(I.Pos, BoxSize, iter->Hsml, iter->symmetric, current)) {
            /* in case the node can be discarded */
            no = current->sibling;
            continue;
        }

        if(mode == TREEWALK_TOPTREE) {
            if(current->f.ChildType == PSEUDO_NODE_TYPE) {
                /* Export the pseudo particle*/
                if(-1 == export_particle(current->s.suns[0]))
                    return -1;
                /* Move sideways*/
                no = current->sibling;
                continue;
            }
            /* Only walk toptree nodes here*/
            if(current->f.TopLevel && !current->f.InternalTopLevel) {
                no = current->sibling;
                continue;
            }
        }
        else {
            /* Node contains relevant particles, add them.*/
            if(current->f.ChildType == PARTICLE_NODE_TYPE) {
                int i;
                int * suns = current->s.suns;
                for (i = 0; i < current->s.noccupied; i++) {
                    ngblist[numcand++] = suns[i];
                }
                /* Move sideways*/
                no = current->sibling;
                continue;
            }
            else if(current->f.ChildType == PSEUDO_NODE_TYPE) {
                /* pseudo particle */
                if(mode == TREEWALK_GHOSTS) {
                    endrun(12312, "Secondary for particle %d from node %d found pseudo at %d.\n", target, startnode, no);
                } else {
                    /* This has already been evaluated with the toptree. Move sideways.*/
                    no = current->sibling;
                    continue;
                }
            }
        }
        /* ok, we need to open the node */
        no = current->s.suns[0];
    }

    return numcand;
}

/**********
 *
 * This particular TreeWalkVisitFunction that uses the nbgiter memeber of
 * The TreeWalk object to iterate over the neighbours of a Query.
 *
 * All Pairwise interactions are implemented this way.
 *
 * Note: Short range gravity is not based on pair enumeration.
 * We may want to port it over and see if gravtree.c receives any speed up.
 *
 * Required fields in TreeWalk: ngbiter, ngbiter_type_elsize.
 *
 * Before the iteration starts, ngbiter is called with iter->base.other == -1.
 * The callback function shall initialize the interator with Hsml, mask, and symmetric.
 *
 *****/
template <typename NgbIterType, typename QueryType, typename ResultType>
int LocalTreeWalk<NgbIterType, QueryType, ResultType>::visit(const QueryType& input, ResultType * output)
{
    NgbIterType iter(input);
    /* Check whether the tree contains the particles we are looking for*/
    if((tree->mask & iter.mask) != iter.mask)
        endrun(5, "Treewalk for particles with mask %d but tree mask is only %d overlap %d.\n", iter.mask, tree->mask, tree->mask & iter.mask);
    /* If symmetric, make sure we did hmax first*/
    if(iter.symmetric == NGB_TREEFIND_SYMMETRIC && !tree->hmax_computed_flag)
        endrun(3, "%s tried to do a symmetric treewalk without computing hmax!\n", ev_label);
    const double BoxSize = tree->BoxSize;

    int64_t ninteractions = 0;
    int inode = 0;

    for(inode = 0; inode < NODELISTLENGTH && input->NodeList[inode] >= 0; inode++)
    {
        int numcand = ngb_treefind_threads(input, &iter, input->NodeList[inode]);
        /* Export buffer is full end prematurally */
        if(numcand < 0)
            return numcand;

        /* If we are here, export is successful. Work on this particle -- first
         * filter out all of the candidates that are actually outside. */
        int numngb;

        for(numngb = 0; numngb < numcand; numngb ++) {
            int other = ngblist[numngb];

            /* Skip garbage*/
            if(Part[other].IsGarbage)
                continue;
            /* In case the type of the particle has changed since the tree was built.
             * Happens for wind treewalk for gas turned into stars on this timestep.*/
            if(!((1<<Part[other].Type) & iter.mask)) {
                continue;
            }

            double dist;

            if(iter.symmetric == NGB_TREEFIND_SYMMETRIC) {
                dist = DMAX(Part[other].Hsml, iter.Hsml);
            } else {
                dist = iter.Hsml;
            }

            double r2 = 0;
            int d;
            double h2 = dist * dist;
            for(d = 0; d < 3; d ++) {
                /* the distance vector points to 'other' */
                iter.dist[d] = NEAREST(input->Pos[d] - Part[other].Pos[d], BoxSize);
                r2 += iter.dist[d] * iter.dist[d];
                if(r2 > h2) break;
            }
            if(r2 > h2) continue;

            /* update the iter and call the iteration function*/
            iter.r2 = r2;
            iter.r = sqrt(r2);
            iter.other = other;
            iter.ngbiter(input, output);
        }

        ninteractions += numngb;
    }

    treewalk_add_counters(ninteractions);

    return 0;
}

/*****
 * Variant of ngbiter that doesn't use the Ngblist.
 * The ngblist is generally preferred for memory locality reasons.
 * Use this variant if the evaluation
 * wants to change the search radius, such as for knn algorithms
 * or some density code. Don't use it if the treewalk modifies other particles.
 * */
 template <typename NgbIterType, typename QueryType, typename ResultType>
 int LocalTreeWalk<NgbIterType, QueryType, ResultType>::visit_nolist_ngbiter(const QueryType& input, ResultType * output)
{
    NgbIterType iter(input);

    int64_t ninteractions = 0;
    int inode;
    for(inode = 0; inode < NODELISTLENGTH && input->NodeList[inode] >= 0; inode++)
    {
        int no = input->NodeList[inode];
        const double BoxSize = tree->BoxSize;

        while(no >= 0)
        {
            struct NODE *current = &tree->Nodes[no];

            /* When walking exported particles we start from the encompassing top-level node,
            * so if we get back to a top-level node again we are done.*/
            if(mode == TREEWALK_GHOSTS) {
                /* The first node is always top-level*/
                if(no > tree->lastnode)
                    endrun(7, "Node is after lastnode. no %d lastnode %ld start %d first %ld\n", no, tree->lastnode, input->NodeList[inode], tree->firstnode);
                if(current->f.TopLevel && no != input->NodeList[inode]) {
                    /* we reached a top-level node again, which means that we are done with the branch */
                    break;
                }
            }

            /* Cull the node */
            if(0 == cull_node(input->Pos, BoxSize, iter->Hsml, iter->symmetric, current)) {
                /* in case the node can be discarded */
                no = current->sibling;
                continue;
            }
            if(mode == TREEWALK_TOPTREE) {
                if(current->f.ChildType == PSEUDO_NODE_TYPE) {
                    /* Export the pseudo particle*/
                    if(-1 == export_particle(current->s.suns[0]))
                        return -1;
                    /* Move sideways*/
                    no = current->sibling;
                    continue;
                }
                /* Only walk toptree nodes here*/
                if(current->f.TopLevel && !current->f.InternalTopLevel) {
                    no = current->sibling;
                    continue;
                }
            }
            /* Node contains relevant particles, add them.*/
            else {
                if(current->f.ChildType == PARTICLE_NODE_TYPE) {
                    int i;
                    int * suns = current->s.suns;
                    for (i = 0; i < current->s.noccupied; i++) {
                        /* Now evaluate a particle for the list*/
                        int other = suns[i];
                        /* Skip garbage*/
                        if(Part[other].IsGarbage)
                            continue;
                        /* In case the type of the particle has changed since the tree was built.
                        * Happens for wind treewalk for gas turned into stars on this timestep.*/
                        if(!((1<<Part[other].Type) & iter->mask))
                            continue;

                        double dist = iter.Hsml;
                        double r2 = 0;
                        int d;
                        double h2 = dist * dist;
                        for(d = 0; d < 3; d ++) {
                            /* the distance vector points to 'other' */
                            iter.dist[d] = NEAREST(input->Pos[d] - Part[other].Pos[d], BoxSize);
                            r2 += iter.dist[d] * iter.dist[d];
                            if(r2 > h2) break;
                        }
                        if(r2 > h2) continue;

                        /* update the iter and call the iteration function*/
                        iter.r2 = r2;
                        iter.other = other;
                        iter.r = sqrt(r2);
                        iter.ngbiter(input, output);
                        ninteractions++;
                    }
                    /* Move sideways*/
                    no = current->sibling;
                    continue;
                }
                else if(current->f.ChildType == PSEUDO_NODE_TYPE) {
                    /* pseudo particle */
                    if(mode == TREEWALK_GHOSTS) {
                        endrun(12312, "Secondary for particle %d from node %d found pseudo at %d.\n", target, input->NodeList[inode], no);
                    } else {
                        /* This has already been evaluated with the toptree. Move sideways.*/
                        no = current->sibling;
                        continue;
                    }
                }
            }
            /* ok, we need to open the node */
            no = current->s.suns[0];
        }
    }

    treewalk_add_counters(ninteractions);

    return 0;
}
