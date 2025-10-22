#!/bin/bash

if [ "$#" -lt 3 ] ; then
  echo
  echo "Usage:  build_release_package.sh <app_version> <arm|intel> <dir_to_fastsurfer> [<url_to_freesurfer>]"
  echo
  exit
fi

# cd into directory with this script 
dir=${0%/*}
if [ -d "$dir" ]; then
  cd "$dir"
fi 

VERSION=$1 # version of the project
ARCH_TYPE=$2 # chip architecture - arm or intel
DIR_TO_FASTSURFER=$3 # directory to fastsurfer
if [ "$#" -gt 3 ]; then
    URL_TO_FREESURFER=$4 # freesurfer install url 
fi

ARCH_TYPE_NAME="arm64"
if [ "$ARCH_TYPE" = "intel" ]; then
    ARCH_TYPE_NAME="x86_64"
fi

PACKAGE_NAME=FastSurfer$VERSION-macos-darwin_${ARCH_TYPE_NAME} # name of the package displayed in the installer
ID="ord.deep-mi.FastSurfer.${VERSION}_${ARCH_TYPE_NAME}" # package identifier (f.e. com.mycompany.productid)
INSTALLATION_DIR="/Applications" # install location for the content of the package
OUTPUT_PKG="raw_package/$PACKAGE_NAME.pkg" # raw package file to be created
INSTALLER_PKG="installer/$PACKAGE_NAME.pkg" # installer to be created

# create temporary folder to package and copy FastSurfer over
STAGED_DIR="FastSurferPackageContent"
FASTSURFER_TO_PACKAGE="$STAGED_DIR/FastSurfer$VERSION"
mkdir $STAGED_DIR
rsync -av --progress $DIR_TO_FASTSURFER/ $FASTSURFER_TO_PACKAGE \
      --exclude requirements.txt \
      --exclude requirements.cpu.txt \
      --exclude Docker \
      --exclude Singularity \
      --exclude tools

# install freesurfer into temp folder
if [ "$#" -gt 3 ]; then
    ../install_fs_pruned.sh $STAGED_DIR --upx --url $URL_TO_FREESURFER
else
    ../install_fs_pruned.sh $STAGED_DIR --upx
fi

SCRIPTS_DIR="./scripts" # directory with scripts executed during installation process (f.e. preinsatll postinstall)

# substitute values in postinstall script
PATH_TO_FASTSURFER="$INSTALLATION_DIR/FastSurfer$VERSION"
cp $SCRIPTS_DIR/postinstall.template $SCRIPTS_DIR/postinstall

sed -i '' -e "s|<fastsurfer_home_dir>|${PATH_TO_FASTSURFER}|g" $SCRIPTS_DIR/postinstall
if [ "$ARCH_TYPE" = "arm"      ]; then
    sed -i '' -e "s|<homebrew_dir>|/opt/homebrew|g" $SCRIPTS_DIR/postinstall
else
    sed -i '' -e "s|<homebrew_dir>|/usr/local|g" $SCRIPTS_DIR/postinstall
fi

# assemble resources
mkdir resources
cp $DIR_TO_FASTSURFER/doc/images/fastsurfer.png resources/
cp $DIR_TO_FASTSURFER/doc/overview/MACOS.md resources/
cp $DIR_TO_FASTSURFER/LICENSE resources/LICENSE.txt

# create fastsurfer applet
cp FastSurfer.py.template FastSurfer.py
sed -i '' -e "s|<fastsurfer>|FastSurfer${VERSION}|g" FastSurfer.py

cp macos_setup_fastsurfer.sh.template macos_setup_fastsurfer.sh
sed -i '' -e "s|<fastsurfer>|FastSurfer${VERSION}|g" macos_setup_fastsurfer.sh
if [ "$ARCH_TYPE" = "arm" ]; then
    sed -i '' -e "s|<mps_fallback_value>|1|g" macos_setup_fastsurfer.sh
else
    sed -i '' -e "s|<mps_fallback_value>|0|g" macos_setup_fastsurfer.sh
fi
mv macos_setup_fastsurfer.sh $FASTSURFER_TO_PACKAGE/

python3.10 setup.py py2app --iconfile resources/fastsurfer.png
mv dist/FastSurfer.app $STAGED_DIR/

rm -f FastSurfer.py
chmod -R 755 $STAGED_DIR/*

# create raw package
mkdir raw_package
pkgbuild \
    --root $STAGED_DIR \
    --version $VERSION \
    --identifier $ID \
    --install-location $INSTALLATION_DIR \
    --scripts $SCRIPTS_DIR \
    $OUTPUT_PKG

rm -f $SCRIPTS_DIR/postinstall

# create distribution file template based on provided package
RESOURCES="./resources"
DISTRIBUTION_FILE="resources/distribution.xml"
productbuild --synthesize --package $OUTPUT_PKG $DISTRIBUTION_FILE

# edit the distribution file
# set title to package name (f.e. package_name.pkg -> <title>package_name</title>)
python3.10 edit_distribution.py --file "$DISTRIBUTION_FILE" --title "$PACKAGE_NAME"

# create installer package
mkdir installer
productbuild \
    --distribution $DISTRIBUTION_FILE \
    --resources $RESOURCES \
    --package-path raw_package \
    $INSTALLER_PKG

# get rid of temporary folder
rm -rf $STAGED_DIR
# rm -rf resources
