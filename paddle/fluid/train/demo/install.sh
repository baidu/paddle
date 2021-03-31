set -e

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
PADDLE_INSTALL="$ROOT/build"

PADDLE_INSTALL_INC="$PADDLE_INSTALL/include"
# INC=$(find . -type f -name "*.h" -exec cp --parents \{\} $PADDLE_INSTALL_INC \;)
INC=$(find . -type f -name "*.h" -exec dirname {} \;)
echo $INC
