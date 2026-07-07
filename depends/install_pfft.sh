#!/bin/sh -e

PREFIX="$1"
shift
OPTIMIZE="$*"
OPTIMIZE1="$*"
echo "Optimization for double" ${OPTIMIZE}

PFFT_VERSION=1.0.8-alpha3-fftw3-2don2d
TMP="tmp-pfft-$PFFT_VERSION"
LOGFILE="build.log"

mkdir -p $TMP
ROOT=`dirname $0`
if ! [ -f $ROOT/pfft-$PFFT_VERSION.tar.gz ]; then
wget https://github.com/rainwoodman/pfft/releases/download/$PFFT_VERSION/pfft-$PFFT_VERSION.tar.gz \
    -O $ROOT/pfft-$PFFT_VERSION.tar.gz 
fi

gzip -dc $ROOT/pfft-$PFFT_VERSION.tar.gz | tar xf - -C $TMP
cd $TMP

(
mkdir -p double;cd double

../pfft-${PFFT_VERSION}/configure --prefix=$PREFIX --disable-shared --enable-static --enable-openmp \
--disable-fortran --disable-dependency-tracking --disable-doc --enable-mpi ${OPTIMIZE} &&
make -j 8   &&
make install && echo "PFFT_DONE"
) 2>&1 > ${LOGFILE}.double

if ! grep PFFT_DONE ${LOGFILE}.double > /dev/null; then
    tail ${LOGFILE}.double
    exit 1
fi

# Single precision fftw (libfftw3f) is needed at link time by the heffte
# FFTW backend, even though the code only performs double precision FFTs.
# -Wno-error=incompatible-pointer-types: gcc >= 14 upgrades an old benign
# double*/float* mismatch in the bundled fftw bench code to an error.
(
mkdir -p single;cd single

CFLAGS="$CFLAGS -Wno-error=incompatible-pointer-types" \
../pfft-${PFFT_VERSION}/configure --prefix=$PREFIX --disable-shared --enable-static --enable-openmp \
--enable-single --disable-fortran --disable-dependency-tracking --disable-doc --enable-mpi ${OPTIMIZE1} &&
make -j 8   &&
make install && echo "PFFT_DONE"
) 2>&1 > ${LOGFILE}.single

if ! grep PFFT_DONE ${LOGFILE}.single > /dev/null; then
    tail ${LOGFILE}.single
    exit 1
fi
