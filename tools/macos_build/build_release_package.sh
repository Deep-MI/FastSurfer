#!/bin/bash

if [[ "$#" -lt 1 ]] || { [[ "$1" != "arm" ]] && [[ "$1" != "intel" ]] ; } ; then
  echo
  echo "Usage:  build_release_package.sh <arm|intel> [--fs-download-cache path] [--fs-pruned-cache-dir dir]"
  echo "                                             [--py2app-venv dir]"
  echo
  echo "--fs-download-cache points at a file path for the raw FreeSurfer tarball: if it already"
  echo "  exists there (e.g. from a prior, interrupted local run), it is reused instead of"
  echo "  downloading again; if not, the download is saved there for a later run to reuse."
  echo "  (default: \$FS_DOWNLOAD_CACHE, if set)"
  echo "--fs-pruned-cache-dir points at a directory for the pruned FreeSurfer install: if a valid"
  echo "  one is already there, the whole download+prune step is skipped."
  echo "  (default: \$FS_PRUNED_CACHE_DIR, if set)"
  echo "--py2app-venv points at a venv to create (if missing) and reuse for the py2app packaging"
  echo "  step, isolated from your normal dev venv, whose extra installed packages (matplotlib,"
  echo "  etc.) py2app's dependency scanner can otherwise trip over."
  echo "  (default: \$PY2APP_VENV, or tools/macos_build/.venv-py2app)"
  echo
  exit
fi
ARCH_TYPE=$1 # chip architecture - arm or intel
shift

fs_download_cache="$FS_DOWNLOAD_CACHE"
fs_pruned_cache_dir="$FS_PRUNED_CACHE_DIR"
py2app_venv="$PY2APP_VENV"
while [[ "$#" -ge 1 ]] ; do
  case "$1" in
  --fs-download-cache) fs_download_cache=$2 ; shift ; shift ;;
  --fs-pruned-cache-dir) fs_pruned_cache_dir=$2 ; shift ; shift ;;
  --py2app-venv) py2app_venv=$2 ; shift ; shift ;;
  *) echo "Invalid argument $1" ; exit 1 ;;
  esac
done

if [[ -z "${BASH_SOURCE[0]}" ]]; then THIS_SCRIPT="$0"
else THIS_SCRIPT="${BASH_SOURCE[0]}"
fi
build_dir=$(dirname "$THIS_SCRIPT")
tools_dir=$(dirname "$build_dir")
py2app_venv="${py2app_venv:-$build_dir/.venv-py2app}"

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

# install pruned freesurfer (not a full install, so nested inside FastSurfer's own directory
# rather than the canonical /Applications/freesurfer, to avoid colliding with a real FreeSurfer install).
# --fs-pruned-cache-dir, if set (e.g. by CI, restored via actions/cache), lets install_fs_pruned.sh
# skip the download+prune entirely when a matching install is already there; either way, the result
# is copied into the staged package content, so the packaging steps below don't need to know about caching.
fs_pruned_where="${fs_pruned_cache_dir:-$FASTSURFER_TO_PACKAGE}"
download_cache_args=()
if [[ -n "$fs_download_cache" ]] ; then download_cache_args=(--fs-download-cache "$fs_download_cache") ; fi
"$tools_dir/build/install_fs_pruned.sh" "$fs_pruned_where" --url "$URL_TO_FREESURFER" --name fs-pruned "${download_cache_args[@]}"

if [[ ! -f "$fs_pruned_where/fs-pruned/build-stamp.txt" ]]
then
  echo "FreeSurfer install was unsuccessful!"
  exit 1
fi

if [[ -n "$FS_PRUNED_CACHE_DIR" ]]
then
  mkdir -p "$FASTSURFER_TO_PACKAGE"
  # remove any stale fs-pruned first: cp -R copies *into* an existing destination dir instead of
  # replacing it, which would silently nest a leftover from an interrupted prior build
  rm -rf "$FASTSURFER_TO_PACKAGE/fs-pruned"
  cp -R "$fs_pruned_where/fs-pruned" "$FASTSURFER_TO_PACKAGE/"
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
# MACOS.md is no longer in the sphinx docs toctree, but is still used verbatim as the plain-text
# conclusion screen of the installer (see edit_distribution.py), so keep it around
cp "$FASTSURFER_HOME/doc/overview/MACOS.md" "$RESOURCES_DIR"
cp "$FASTSURFER_HOME/LICENSE" "$RESOURCES_DIR/LICENSE.txt"

# create fastsurfer applet
sed -e "s|<fastsurfer>|FastSurfer${VERSION}|g" \
    < "$build_dir/FastSurfer.py.template" \
    > "$build_dir/FastSurfer.py"

sed -e "s|<fastsurfer>|FastSurfer${VERSION}|g" \
    -e "s|<python_version>|${PYTHON_VERSION}|g" \
    < "$build_dir/macos_setup_fastsurfer.sh.template" \
    > "$build_dir/macos_setup_fastsurfer.sh"

mv "$build_dir/macos_setup_fastsurfer.sh" "$FASTSURFER_TO_PACKAGE/"

# isolated venv for py2app, kept separate from any dev venv: py2app's dependency scanner walks the
# whole environment it runs in, and unrelated packages there (e.g. matplotlib) can make it fail
if [[ ! -x "$py2app_venv/bin/python3" ]]
then
  echo "Creating isolated venv for py2app at $py2app_venv ..."
  python3 -m venv "$py2app_venv"
  "$py2app_venv/bin/python3" -m pip install --upgrade pip
  "$py2app_venv/bin/python3" -m pip install py2app
fi

pushd "$build_dir" || exit 1
"$py2app_venv/bin/python3" "setup.py" py2app --iconfile "${RESOURCES_DIR:$((${#build_dir} + 1))}/fastsurfer.png"
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
