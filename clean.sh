#!/usr/bin/env bash
#
# Remove build artifacts produced by build.sh / CMake.
#
# By default this cleans both the main project and the Thirdparty
# dependencies (DBoW2, g2o, Sophus). graphite has no separate build tree --
# it is pulled in with add_subdirectory(), so it lives inside build/.
#
# Usage:
#   ./clean.sh                 clean main project + Thirdparty
#   ./clean.sh --main-only     leave Thirdparty builds alone
#   ./clean.sh --thirdparty    clean only Thirdparty
#   ./clean.sh --vocab         also delete the uncompressed Vocabulary/ORBvoc.txt
#   ./clean.sh --dry-run       print what would be removed, delete nothing
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

CLEAN_MAIN=1
CLEAN_THIRDPARTY=1
CLEAN_VOCAB=0
DRY_RUN=0

while [ $# -gt 0 ]; do
    case "$1" in
        --main-only)   CLEAN_THIRDPARTY=0 ;;
        --thirdparty)  CLEAN_MAIN=0 ;;
        --vocab)       CLEAN_VOCAB=1 ;;
        --dry-run|-n)  DRY_RUN=1 ;;
        -h|--help)     sed -n '2,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)             echo "clean.sh: unknown option '$1' (try --help)" >&2; exit 1 ;;
    esac
    shift
done

# Remove a path if it exists, relative to the repo root.
remove() {
    local target="$1"
    [ -e "$target" ] || return 0
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "  would remove  $target"
    else
        echo "  removing      $target"
        rm -rf -- "$target"
    fi
}

# Delete the compiled example binaries. CMake drops them straight into the
# source tree (see the CMAKE_RUNTIME_OUTPUT_DIRECTORY lines in CMakeLists.txt),
# next to the .sh/.py/.yaml files that belong to the repo. Every binary has an
# extensionless name, so that is what we match on.
remove_example_binaries() {
    local dir
    for dir in Examples Examples_old; do
        [ -d "$dir" ] || continue
        while IFS= read -r -d '' bin; do
            remove "$bin"
        done < <(find "$dir" -type f -executable ! -name '*.*' -print0 | sort -z)
    done
}

if [ "$CLEAN_MAIN" -eq 1 ]; then
    echo "Cleaning main build ..."
    remove build              # includes build/_deps (googletest) and graphite
    remove lib
    remove Examples/ROS/ORB_SLAM3/build
    remove_example_binaries
fi

if [ "$CLEAN_THIRDPARTY" -eq 1 ]; then
    echo "Cleaning Thirdparty ..."
    remove Thirdparty/DBoW2/build
    remove Thirdparty/DBoW2/lib
    remove Thirdparty/g2o/build
    remove Thirdparty/g2o/lib
    remove Thirdparty/g2o/config.h    # generated from config.h.in by cmake
    remove Thirdparty/Sophus/build
fi

if [ "$CLEAN_VOCAB" -eq 1 ]; then
    echo "Cleaning vocabulary ..."
    remove Vocabulary/ORBvoc.txt      # re-extracted from the .tar.gz by build.sh
fi

if [ "$DRY_RUN" -eq 1 ]; then
    echo "Dry run -- nothing was deleted."
else
    echo "Done. Rebuild with ./build.sh"
fi
