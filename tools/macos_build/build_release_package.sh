#!/bin/bash

if [[ "$#" != 1 ]] || { [[ "$1" != "arm" ]] && [[ "$1" != "intel" ]] ; } ; then
  echo
  echo "Usage:  build_release_package.sh <arm|intel>"
  echo
  exit
fi

if [[ -z "${BASH_SOURCE[0]}" ]]; then THIS_SCRIPT="$0"
else THIS_SCRIPT="${BASH_SOURCE[0]}"
fi
build_dir=$(dirname "$THIS_SCRIPT")
tools_dir=$(dirname "$build_dir")

FASTSURFER_HOME=$(dirname "$tools_dir") # directory to fastsurfer
# version of the project
VERSION=$(python3 "$tools_dir/read_toml.py" --file "$FASTSURFER_HOME/pyproject.toml" --key project.version)
VERSION_NO_DOTS=${VERSION//./}
#version of the freesurfer
FREESURFER_VERSION=$(python3 "$tools_dir/read_toml.py" --file "$FASTSURFER_HOME/pyproject.toml" --key tool.freesurfer.version)
# freesurfer install url
URL_TO_FREESURFER_TEMP=$(python3 "$tools_dir/read_toml.py" --file "$FASTSURFER_HOME/pyproject.toml" --key tool.freesurfer.urls.macOS)
sub="{version}"
URL_TO_FREESURFER="${URL_TO_FREESURFER_TEMP//$sub/$FREESURFER_VERSION}"
ARCH_TYPE=$1 # chip architecture - arm or intel

ARCH_TYPE_NAME="arm64"
if [[ "$ARCH_TYPE" = "intel" ]] ; then ARCH_TYPE_NAME="x86_64" ; fi

RESOURCES_DIR="$build_dir/resources"
# name of the package displayed in the installer
PACKAGE_NAME=FastSurfer$VERSION_NO_DOTS-macos-darwin_${ARCH_TYPE_NAME}
# package identifier (f.e. com.mycompany.productid)
ID="org.deep-mi.FastSurfer.${VERSION_NO_DOTS}_${ARCH_TYPE_NAME}"
# install location for the content of the package
INSTALLATION_DIR="/Applications"
# raw package file to be created
OUTPUT_PKG="$build_dir/raw_package/$PACKAGE_NAME.pkg"
# installer to be created
INSTALLER_PKG="$build_dir/installer/$PACKAGE_NAME.pkg"

# create temporary folder to package and copy FastSurfer over
STAGED_DIR="$build_dir/FastSurferPackageContent"
FASTSURFER_TO_PACKAGE="$STAGED_DIR/FastSurfer$VERSION"
mkdir -p "$STAGED_DIR"
rsync -av --progress "$FASTSURFER_HOME/" "$FASTSURFER_TO_PACKAGE" \
      --exclude requirements.txt \
      --exclude requirements.cpu.txt \
      --exclude tools \
      --exclude .git

# install freesurfer into temp folder
"$tools_dir/build/install_fs_pruned.sh" "$STAGED_DIR" --upx --url "$URL_TO_FREESURFER"

if [[ ! -d "$STAGED_DIR/freesurfer" ]]
then
  echo "FreeSurfer install was unsuccessful!"
  exit 1
fi

SCRIPTS_DIR="$tools_dir/macos_build/scripts" # directory with scripts executed during installation process (f.e. preinstall postinstall)
PYTHON_VERSION_TEMP=$(python3 "$tools_dir/read_toml.py" --file "$FASTSURFER_HOME/pyproject.toml" --key project.requires-python)
PYTHON_VERSION="${PYTHON_VERSION_TEMP#>=}"

# substitute values in postinstall script
PATH_TO_FASTSURFER="$INSTALLATION_DIR/FastSurfer$VERSION"
HOMEBREW_DIR=$([[ "$ARCH_TYPE" = "arm" ]] && echo "/opt/homebrew" || echo "/usr/local")

sed -e "s|<fastsurfer_home_dir>|${PATH_TO_FASTSURFER}|g" \
    -e "s|<python_version>|${PYTHON_VERSION}|g" \
    -e "s|<homebrew_dir>|$HOMEBREW_DIR|g" \
    < "$SCRIPTS_DIR/postinstall.sh.template" \
    > "$SCRIPTS_DIR/postinstall"
# copy link_fs script (do not keep double copies, so delete after build)
cp "$tools_dir/build/link_fs.sh" "$SCRIPTS_DIR/link_fs.sh"

chmod +x "$SCRIPTS_DIR/postinstall"

# assemble resources
mkdir -p "$RESOURCES_DIR"
cp "$FASTSURFER_HOME/doc/images/fastsurfer.png" "$RESOURCES_DIR"
cp "$FASTSURFER_HOME/doc/overview/MACOS.md" "$RESOURCES_DIR"
cp "$FASTSURFER_HOME/LICENSE" "$RESOURCES_DIR/LICENSE.txt"

# create fastsurfer applet
sed -e "s|<fastsurfer>|FastSurfer${VERSION}|g" \
    < "$build_dir/FastSurfer.py.template" \
    > "$build_dir/FastSurfer.py"

MPS_FALLBACK_VALUE=$([[ "$ARCH_TYPE" = "arm" ]] && echo 1 || echo 0)
sed -e "s|<fastsurfer>|FastSurfer${VERSION}|g" \
    -e "s|<python_version>|${PYTHON_VERSION}|g" \
    -e "s|<mps_fallback_value>|${MPS_FALLBACK_VALUE}|g" \
    < "$build_dir/macos_setup_fastsurfer.sh.template" \
    > "$build_dir/macos_setup_fastsurfer.sh"

mv "$build_dir/macos_setup_fastsurfer.sh" "$FASTSURFER_TO_PACKAGE/"

pushd "$build_dir" || exit 1
python3 "setup.py" py2app --iconfile "${RESOURCES_DIR:$((${#build_dir} + 1))}/fastsurfer.png"
popd || exit 1
mv "$build_dir/dist/FastSurfer.app" "$STAGED_DIR/FastSurfer$VERSION.app"

rm -f "$build_dir/FastSurfer.py"
chmod -R 755 "$STAGED_DIR"/*

# create raw package
mkdir -p "$build_dir/raw_package"
pkgbuild \
    --root "$STAGED_DIR" \
    --version "$VERSION" \
    --identifier "$ID" \
    --install-location "$INSTALLATION_DIR" \
    --scripts "$SCRIPTS_DIR" \
    "$OUTPUT_PKG"

rm -f "$SCRIPTS_DIR/postinstall"

# create distribution file template based on provided package
DISTRIBUTION_FILE="$RESOURCES_DIR/distribution.xml"

productbuild --synthesize --package "$OUTPUT_PKG" "$DISTRIBUTION_FILE"

# edit the distribution file
# set title to package name (f.e. package_name.pkg -> <title>package_name</title>)
python3 "$build_dir/edit_distribution.py" --file "$DISTRIBUTION_FILE" --title "$PACKAGE_NAME"

# create installer package
mkdir -p "$build_dir/installer"
productbuild \
    --distribution "$DISTRIBUTION_FILE" \
    --resources "$RESOURCES_DIR" \
    --package-path "$build_dir/raw_package" \
    "$INSTALLER_PKG"

# get rid of temporary folder
rm -rf "$STAGED_DIR" "$RESOURCES_DIR" "$build_dir/dist" "$build_dir/build"
# remove the previously copied link_fs.sh script
rm "$SCRIPTS_DIR/link_fs.sh"
