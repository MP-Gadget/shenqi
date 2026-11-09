/*Simple test for the tree building functions in the C++ ForceTree interface (forcetree2.h)*/
#define BOOST_TEST_MODULE forcetree2

#include "booststub.h"

#include <vector>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>

#include <libgadget/utils/system.h>
#include <libgadget/forcetree2.h>
#include <libgadget/partmanager.h>
#include <libgadget/slotsmanager.h>
#include <libgadget/domain.h>
#include <libgadget/walltime.h>

static struct ClockTable Clocks;

/* The tree internals (Nodes, numnodes, maxnodes) are protected.
 * This wrapper exposes them read-only for testing, and inherits the tree constructors.*/
template <typename Tree>
class TestTree : public Tree {
public:
    using Tree::Tree;
    const NODE * nodes() const { return this->Nodes; }
    int64_t get_numnodes() const { return this->numnodes; }
    int64_t get_maxnodes() const { return this->maxnodes; }
    int64_t get_numparticles() const { return this->NumParticles; }
    int64_t get_maxpart() const { return this->MaxPart; }
    /* Time in ms spent in DEBUG validation during construction, to exclude it from build timings*/
    double get_validate_ms() const { return this->validate_time * 1000; }
};

typedef TestTree<ForceTreeMoments> TestTreeMoments;
typedef TestTree<ForceTreeHmax> TestTreeHmax;

/* Results of walking the tree from the root.*/
struct TreeWalkInfo {
    /* Parent node of each particle, as found by walking the tree*/
    std::vector<int> pfather;
    /* Nodes reachable from the root*/
    std::vector<int> reachable;
    /* Number of reachable nodes with no children (only allowed for toplevel nodes)*/
    int nempty = 0;
    /* Number of particle slots left unused in particle nodes*/
    int64_t tot_empty = 0;
};

/* Recursively walk the tree via the suns arrays, validating the geometry
 * and the parent pointers, and recording the parent of each particle.*/
template <typename Tree>
static void walk_node(const Tree& tb, const int no, const int numpart, TreeWalkInfo& info)
{
    const NODE * Nodes = tb.nodes();
    const NODE * nop = &Nodes[no];
    BOOST_TEST_REQUIRE(tb.node_is_node(no));
    info.reachable.push_back(no);

    if(nop->f.ChildType == PARTICLE_NODE_TYPE) {
        BOOST_TEST(nop->s.noccupied >= 0);
        BOOST_TEST(nop->s.noccupied <= NMAXCHILD);
        info.tot_empty += NMAXCHILD - nop->s.noccupied;
        if(nop->s.noccupied == 0) {
            info.nempty++;
            /* Only toplevel nodes should be kept when empty*/
            BOOST_TEST(nop->f.TopLevel);
        }
        for(int j = 0; j < nop->s.noccupied; j++) {
            const int child = nop->s.suns[j];
            BOOST_TEST_REQUIRE(child >= 0);
            BOOST_TEST_REQUIRE(child < numpart);
            /* Each particle should appear exactly once*/
            BOOST_TEST(info.pfather[child] == -1);
            info.pfather[child] = no;
            /* Particle should be inside the node*/
            BOOST_TEST(const_cast<NODE *>(nop)->inside_node(PartManager->Base[child].Pos, 1e-6 * nop->len));
        }
    }
    else if(nop->f.ChildType == NODE_NODE_TYPE) {
        BOOST_TEST(nop->s.noccupied == NODEFULL);
        int nchild = 0;
        int seen_subnode = 0;
        for(int j = 0; j < NMAXCHILD; j++) {
            const int child = nop->s.suns[j];
            /* Empty children are compacted to the end of the array*/
            if(child < 0) {
                for(int jj = j; jj < NMAXCHILD; jj++)
                    BOOST_TEST(nop->s.suns[jj] == -1);
                break;
            }
            nchild++;
            BOOST_TEST_REQUIRE(tb.node_is_node(child));
            const NODE * nchld = &Nodes[child];
            BOOST_TEST(nchld->father == no);
            BOOST_TEST(fabs(nchld->len / nop->len - 0.5) < 1e-4);
            /* Child center should be offset by len/4 in each direction*/
            for(int k = 0; k < 3; k++)
                BOOST_TEST(fabs(fabs(nchld->center[k] - nop->center[k]) - nop->len / 4) < 1e-6 * nop->len);
            /* Each child should be in a distinct octant*/
            const int sub = const_cast<NODE *>(nop)->get_subnode(nchld->center);
            BOOST_TEST(!(seen_subnode & (1 << sub)));
            seen_subnode |= (1 << sub);
            walk_node(tb, child, numpart, info);
        }
        BOOST_TEST(nchild > 0);
    }
    else
        BOOST_TEST_FAIL("Pseudo node " << no << " in a single-task tree");
}

/*This checks that the force tree is valid:
 * that it contains every particle exactly once and that each parent
 * node contains children within the right subnode.*/
template <typename Tree>
static TreeWalkInfo check_tree(const Tree& tb, const int numpart)
{
    TreeWalkInfo info;
    info.pfather.assign(numpart, -1);
    /* Root node*/
    const NODE * Nodes = tb.nodes();
    BOOST_TEST(Nodes[0].father == -1);
    BOOST_TEST(Nodes[0].sibling == -1);
    BOOST_TEST(Nodes[0].f.TopLevel);
    walk_node(tb, 0, numpart, info);

    int64_t nintree = 0;
    for(int i = 0; i < numpart; i++)
        if(info.pfather[i] >= 0)
            nintree++;
    BOOST_TEST(nintree == numpart);
    BOOST_TEST(tb.get_numparticles() == numpart);
    BOOST_TEST(tb.get_maxpart() == PartManager->MaxPart);
    const int64_t nreal = info.reachable.size();
    BOOST_TEST(nreal <= tb.get_numnodes());
    printf("Tree filling factor: %g on %ld nodes (allocated: %ld empty: %d)\n",
           info.tot_empty / (8. * nreal), nreal, tb.get_numnodes(), info.nempty);
    return info;
}

/* Walk the tree in the order used by the treewalk (suns[0] to open a node,
 * sibling to skip it), checking the sibling pointers are valid.*/
template <typename Tree>
static void check_siblings(const Tree& tb, const int64_t nreal)
{
    const NODE * Nodes = tb.nodes();
    int node = 0;
    int counter = 0;
    int sibcntr = 0;
    while(node >= 0) {
        BOOST_TEST_REQUIRE(tb.node_is_node(node));
        const NODE * nop = &Nodes[node];
        const int sib = nop->sibling;
        BOOST_TEST_REQUIRE(sib >= -1);
        BOOST_TEST_REQUIRE(sib < tb.get_numnodes());
        if(sib >= 0) {
            const int sfather = Nodes[sib].father;
            const int father = nop->father;
            /* Our sibling should either be a true sibling, with the same father,
             * or should be the child of one of our ancestors*/
            if(sfather != father) {
                int ances = father;
                while(ances >= 0) {
                    ances = Nodes[ances].father;
                    if(ances == sfather)
                        break;
                }
                BOOST_TEST(ances == sfather);
            }
        }
        else
            sibcntr++;
        counter++;
        BOOST_TEST_REQUIRE(counter <= nreal);

        if(nop->f.ChildType == PARTICLE_NODE_TYPE)
            node = sib;
        else
            node = nop->s.suns[0];
    }
    /* Every reachable node should be visited by the walk*/
    BOOST_TEST(counter == nreal);
    BOOST_TEST(sibcntr < counter/100 + 2);
}

/*This checks that the moments of the force tree are valid:
 * that the mass and center of mass are correct.*/
static void check_moments(const TestTreeMoments& tb, const int numpart, const TreeWalkInfo& info)
{
    const NODE * Nodes = tb.nodes();
    const int64_t numnodes = tb.get_numnodes();
    std::vector<double> mass(numnodes, 0);
    std::vector<double> cofm(3*numnodes, 0);

    /* Accumulate particle moments into all ancestors*/
    for(int i = 0; i < numpart; i++) {
        int fnode = info.pfather[i];
        BOOST_TEST_REQUIRE(fnode >= 0);
        const double pmass = PartManager->Base[i].Mass;
        while(fnode >= 0) {
            BOOST_TEST_REQUIRE(fnode < numnodes);
            mass[fnode] += pmass;
            for(int k = 0; k < 3; k++)
                cofm[3*fnode + k] += pmass * PartManager->Base[i].Pos[k];
            fnode = Nodes[fnode].father;
        }
    }
    int nbad = 0;
    for(const int node : info.reachable) {
        const NODE * nop = &Nodes[node];
        if(fabs(nop->mom.mass - mass[node]) > 0.5) {
            if(nbad++ < 10)
                printf("node %d mass %g / expected %g TL %d DLM %d ITL %d type %d\n",
                    node, nop->mom.mass, mass[node], nop->f.TopLevel,
                    nop->f.DependsOnLocalMass, nop->f.InternalTopLevel, nop->f.ChildType);
        }
        BOOST_TEST(fabs(nop->mom.mass - mass[node]) < 0.5);
        for(int k = 0; k < 3; k++) {
            BOOST_TEST(nop->mom.cofm[k] <= PartManager->BoxSize);
            BOOST_TEST(nop->mom.cofm[k] >= 0);
            if(mass[node] > 0)
                BOOST_TEST(fabs(nop->mom.cofm[k] - cofm[3*node+k] / mass[node]) < 1e-6 * PartManager->BoxSize);
            else
                BOOST_TEST(nop->mom.cofm[k] == nop->center[k]);
        }
    }
    /* Local mass flag is set on the topleaf*/
    BOOST_TEST(Nodes[0].f.DependsOnLocalMass);
}

/* Set up the particle properties common to all tests and sort them.*/
static void setup_particles(const int numpart)
{
    #pragma omp parallel for
    for(int i=0; i<numpart; i++) {
        PartManager->Base[i].Mass = 1;
        PartManager->Base[i].PI = 0;
        PartManager->Base[i].IsGarbage = 0;
        PartManager->Base[i].Swallowed = 0;
        PartManager->Base[i].TopLeaf = 0;
    }
    PartManager->MaxPart = numpart;
    PartManager->NumPart = numpart;
    /*Sort by peano key so this is more realistic*/
    slots_gc_sorted(PartManager, SlotsManager);
}

/* Build a moments tree and check it. The tree is built three ways:
 * from a NULL active list, from an empty active list (meaning all particles)
 * and from an explicit list of all particles. Only the last should not be flagged as a full tree.*/
static void do_tree_test(const int numpart, DomainDecomp * ddecomp)
{
    setup_particles(numpart);
    ActiveParticles EmptyAct = init_empty_active_particles(PartManager);
    /* Explicit active list containing every particle*/
    std::vector<int> actlist(numpart);
    for(int i = 0; i < numpart; i++)
        actlist[i] = i;
    ActiveParticles ListAct = init_empty_active_particles(PartManager);
    ListAct.ActiveParticle = actlist.data();
    ListAct.MaxActiveParticle = numpart;

    struct { const char * name; ActiveParticles * act; bool full; } cases[] = {
        {"NULL", NULL, true},
        {"empty", &EmptyAct, true},
        {"list", &ListAct, false},
    };
    for(const auto& c : cases) {
        /*Time creating the nodes and computing the moments*/
        double start = MPI_Wtime();
        TestTreeMoments tb(*PartManager, ddecomp, c.act, false, "");
        double end = MPI_Wtime();
        double ms = (end - start)*1000 - tb.get_validate_ms();
        BOOST_TEST(tb.full_particle_tree_flag == c.full);
        BOOST_TEST(tb.get_numnodes() < tb.get_maxnodes());
        printf("Active list %s: number of nodes used: %ld. Built tree and moments in %.3g ms (excluding %.3g ms validation). Total mass: %g\n",
               c.name, tb.get_numnodes(), ms, tb.get_validate_ms(), tb.nodes()[0].mom.mass);
        BOOST_TEST(fabs(tb.nodes()[0].mom.mass - numpart) < 0.5);
        TreeWalkInfo info = check_tree(tb, numpart);
        check_siblings(tb, info.reachable.size());
        check_moments(tb, numpart, info);
    }
}

/* Find the hmax value between a node and a particle.*/
static double compute_distance(int i, const NODE * node)
{
    double hmax = 0;
    for(int j = 0; j < 3; j++) {
        /* Compute each direction independently and take the maximum.
            * This is the largest possible distance away from node center within a cube bounding hsml.
            * Note that because Pos - Center < len/2, the maximum value this can have is Hsml.*/
        hmax = DMAX(hmax, fabs(PartManager->Base[i].Pos[j] - node->center[j]) + PartManager->Base[i].Hsml - node->len/2.);
    }
    return hmax;
}

/* Test whether particle is inside node.
 * 0 if outside node, 1 if inside node. */
static int check_inside(int i, const NODE * node)
{
    for(int j = 0; j < 3; j++)
        if (fabs(PartManager->Base[i].Pos[j] - node->center[j]) > node->len/2)
            return 0;
    return 1;
}

/* This checks that the hmax moment in Nodes is correct:
 * that is the distance between each particle and node center is larger than hmax.*/
static void check_hmax(const TestTreeHmax& tb, const int numpart, const TreeWalkInfo& info)
{
    const NODE * Nodes = tb.nodes();
    for(int i=0; i<numpart; i++)
    {
        /* Father array should agree with the tree walk*/
        BOOST_TEST(tb.get_father_particle(i) == info.pfather[i]);
        int j = tb.get_father_particle(i);
        while(j >= 0) {
            /* Test whether particle is in node*/
            BOOST_TEST(check_inside(i, &Nodes[j]));
            /* Test whether hmax is set correctly*/
            BOOST_TEST(compute_distance(i, &Nodes[j]) <= Nodes[j].mom.hmax+1e-5);
            BOOST_TEST(Nodes[j].mom.hmax >= 0);
            j = tb.get_father_node(j);
        }
    }
}

/* Build a gas-only hmax tree. If active is true, all particles are active and
 * hmax is set after the build with update_tree_hmax_father (as in density_postprocess).
 * Otherwise all particles are inactive and hmax is set during the tree build.*/
static double do_tree_mask_hmax_update_test(const int numpart, DomainDecomp * ddecomp, const bool active)
{
    RandTable rnd = set_random_numbers(23, 8192);

    #pragma omp parallel for
    for(int i=0; i<numpart; i++) {
        PartManager->Base[i].Type = 0;
        PartManager->Base[i].Hsml = PartManager->BoxSize/cbrt(numpart) * get_random_number(i, &rnd);
        /* Bin 0 is always active. Bin 2 is inactive at Ti = 1.*/
        PartManager->Base[i].TimeBinHydro = active ? 0 : 2;
        PartManager->Base[i].Ti_drift = 1;
    }
    free_random_numbers(&rnd);
    SlotsManager->info[0].enabled = 0;
    setup_particles(numpart);

    double start = MPI_Wtime();
    TestTreeHmax tb(*PartManager, ddecomp, GASMASK, "");
    double end = MPI_Wtime();
    double ms = (end - start)*1000 - tb.get_validate_ms();
    printf("Built gas tree in %.3g ms (excluding %.3g ms validation)\n", ms, tb.get_validate_ms());
    BOOST_TEST(tb.hmax_computed_flag == false);
    BOOST_TEST(tb.Father);
    BOOST_TEST(tb.nfather == numpart);

    start = MPI_Wtime();
    if(active) {
        #pragma omp parallel for
        for(int i = 0; i < numpart; i++)
            tb.update_tree_hmax_father(i, PartManager->Base[i].Pos, PartManager->Base[i].Hsml);
    }
    /* now compute the hmax recursively */
    tb.force_tree_calc_hmax(ddecomp);
    end = MPI_Wtime();
    ms = (end - start)*1000;
    BOOST_TEST(tb.hmax_computed_flag == true);
    const double roothmax = tb.nodes()[0].mom.hmax;
    printf("Updated hmax (active = %d) in %.3g ms. Root hmax: %g\n", active, ms, roothmax);
    TreeWalkInfo info = check_tree(tb, numpart);
    check_siblings(tb, info.reachable.size());
    check_hmax(tb, numpart, info);
    return roothmax;
}

static void do_hmax_tests(const int numpart, DomainDecomp * ddecomp, const double minroothmax)
{
    const double inactive = do_tree_mask_hmax_update_test(numpart, ddecomp, false);
    BOOST_TEST(inactive >= minroothmax);
    const double active = do_tree_mask_hmax_update_test(numpart, ddecomp, true);
    BOOST_TEST(active >= minroothmax);
    /* Same particles, so both routes should agree*/
    BOOST_TEST(fabs(active - inactive) < 1e-6);
}

/*Make a simple trivial domain for all data on a single processor*/
static void trivial_domain(DomainDecomp * ddecomp)
{
    /* The whole tree goes into one topnode.
     * Set up just enough of the TopNode structure that
     * domain_get_topleaf works*/
    ddecomp->domain_allocated_flag = 1;
    ddecomp->NTopNodes = 1;
    ddecomp->NTopLeaves = 1;
    ddecomp->TopNodes = (struct topnode_data * ) malloc(sizeof(struct topnode_data));
    ddecomp->TopNodes[0].Daughter = -1;
    ddecomp->TopNodes[0].Leaf = 0;
    ddecomp->TopLeaves = (struct topleaf_data * )malloc(sizeof(struct topleaf_data));
    ddecomp->TopLeaves[0].Task = 0;
    /*These are not used*/
    ddecomp->TopNodes[0].StartKey = 0;
    ddecomp->TopNodes[0].Shift = BITS_PER_DIMENSION * 3;
    /*To tell the code we are in serial*/
    ddecomp->Tasks = (struct task_data * ) malloc(sizeof(struct task_data));
    ddecomp->Tasks[0].StartLeaf = 0;
    ddecomp->Tasks[0].EndLeaf = 1;
}

static DomainDecomp setup_tree(void) {
    /*Set up the important parts of the All structure.*/
    /*Particles should not be outside this*/
    memset(PartManager, 0, sizeof(PartManager[0]));
    memset(SlotsManager, 0, sizeof(SlotsManager[0]));
    PartManager->BoxSize = 8;
    /*Set up the top-level domain grid*/
    DomainDecomp ddecomp;
    trivial_domain(&ddecomp);
    walltime_init(&Clocks);
    return ddecomp;
}

static void teardown_tree(DomainDecomp ddecomp)
{
    free(ddecomp.TopNodes);
    free(ddecomp.TopLeaves);
    free(ddecomp.Tasks);
}

BOOST_AUTO_TEST_CASE(test_rebuild_flat)
{
    DomainDecomp ddecomp = setup_tree();
    /*Set up the particle data*/
    int ncbrt = 128;
    int numpart = ncbrt*ncbrt*ncbrt;
    particle_alloc_memory(PartManager, 8, numpart);
    /* Create a regular grid of particles, all of type 1,
     * in a box 8 kpc across.*/
    #pragma omp parallel for
    for(int i=0; i<numpart; i++) {
        PartManager->Base[i].Type = 1;
        PartManager->Base[i].Pos[0] = (PartManager->BoxSize/ncbrt) * (i/ncbrt/ncbrt);
        PartManager->Base[i].Pos[1] = (PartManager->BoxSize/ncbrt) * ((i/ncbrt) % ncbrt);
        PartManager->Base[i].Pos[2] = (PartManager->BoxSize/ncbrt) * (i % ncbrt);
    }
    PartManager->NumPart = numpart;
    do_tree_test(numpart, &ddecomp);
    do_hmax_tests(numpart, &ddecomp, 0.0584);
    myfree(PartManager->Base);
    teardown_tree(ddecomp);
}

BOOST_AUTO_TEST_CASE(test_rebuild_close)
{
    DomainDecomp ddecomp = setup_tree();
    /*Set up the particle data*/
    int ncbrt = 128;
    int numpart = ncbrt*ncbrt*ncbrt;
    double close = 5000;
    particle_alloc_memory(PartManager, 8, numpart);
    /* Create particles clustered in one place, all of type 1.*/
    #pragma omp parallel for
    for(int i=0; i<numpart; i++) {
        PartManager->Base[i].Type = 1;
        PartManager->Base[i].Pos[0] = 4. + (i/ncbrt/ncbrt)/close;
        PartManager->Base[i].Pos[1] = 4. + ((i/ncbrt) % ncbrt) /close;
        PartManager->Base[i].Pos[2] = 4. + (i % ncbrt)/close;
    }
    PartManager->NumPart = numpart;
    do_tree_test(numpart, &ddecomp);
    do_hmax_tests(numpart, &ddecomp, 0);
    myfree(PartManager->Base);
    teardown_tree(ddecomp);
}

static void do_random_test(boost::random::mt19937 & r, const int numpart, DomainDecomp * ddecomp)
{
    boost::random::uniform_real_distribution<double> dist(0, 1);
    /* Create particles: a uniform background and two clusters, all of type 1,
     * in a box 8 kpc across.*/
    for(int i=0; i<numpart/4; i++) {
        PartManager->Base[i].Type = 1;
        for(int j=0; j<3; j++)
            PartManager->Base[i].Pos[j] = PartManager->BoxSize * dist(r);
    }
    for(int i=numpart/4; i<3*numpart/4; i++) {
        PartManager->Base[i].Type = 1;
        for(int j=0; j<3; j++)
            PartManager->Base[i].Pos[j] = PartManager->BoxSize/2 + PartManager->BoxSize/8 * exp(pow(dist(r)-0.5,2));
    }
    for(int i=3*numpart/4; i<numpart; i++) {
        PartManager->Base[i].Type = 1;
        for(int j=0; j<3; j++)
            PartManager->Base[i].Pos[j] = PartManager->BoxSize*0.1 + PartManager->BoxSize/32 * exp(pow(dist(r)-0.5,2));
    }
    PartManager->NumPart = numpart;
    do_tree_test(numpart, ddecomp);
}

BOOST_AUTO_TEST_CASE(test_rebuild_random)
{
    DomainDecomp ddecomp = setup_tree();
    /*Set up the particle data*/
    int ncbrt = 64;
    auto r = boost::random::mt19937(0);
    int numpart = ncbrt*ncbrt*ncbrt;
    particle_alloc_memory(PartManager, 8, numpart);
    for(int i=0; i<2; i++) {
        do_random_test(r, numpart, &ddecomp);
    }
    do_hmax_tests(numpart, &ddecomp, 0);
    myfree(PartManager->Base);
    teardown_tree(ddecomp);
}
