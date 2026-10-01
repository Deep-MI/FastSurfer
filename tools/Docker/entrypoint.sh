#!/bin/bash --login
# The --login ensures the bash configuration is loaded,
# enabling Conda.

# Enable strict mode.
#set -euo pipefail
# ... Run whatever commands ...

# Temporarily disable strict mode and activate venv:
set +euo pipefail
source /venv/bin/activate

# Re-enable strict mode:
set -euo pipefail

# exec the final command:
"$@"
