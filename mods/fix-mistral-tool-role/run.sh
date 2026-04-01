#!/bin/bash
# Runtime mod: Allow 'user' role after 'tool' in mistral_common validator
#
# The Mistral protocol strictly requires tool -> assistant -> user ordering,
# but OpenAI-compatible clients (e.g. OpenClaw) may send tool -> user directly.
# This patches the validator to accept that pattern.

set -euo pipefail

VALIDATOR_FILE=$(python3 -c "
import mistral_common.protocol.instruct.validator as v, os
print(os.path.abspath(v.__file__))
" 2>/dev/null)

if [ -z "$VALIDATOR_FILE" ] || [ ! -f "$VALIDATOR_FILE" ]; then
    echo "ERROR: Cannot find mistral_common validator.py"
    exit 1
fi

echo "Patching: $VALIDATOR_FILE"

# Check if already patched
if grep -q 'Roles.user.*# allow user after tool' "$VALIDATOR_FILE" 2>/dev/null; then
    echo "  Already patched — skipping"
    exit 0
fi

python3 << 'PYEOF'
import sys

filepath = sys.argv[1] if len(sys.argv) > 1 else ""
if not filepath:
    import mistral_common.protocol.instruct.validator as v, os
    filepath = os.path.abspath(v.__file__)

with open(filepath, 'r') as f:
    content = f.read()

old = '                    expected_roles = {Roles.assistant, Roles.tool}'
new = '                    expected_roles = {Roles.assistant, Roles.tool, Roles.user}  # allow user after tool'

if old not in content:
    print("  WARNING: Could not find tool role validation — may already be patched or changed")
    sys.exit(0)

content = content.replace(old, new)

with open(filepath, 'w') as f:
    f.write(content)

print("  Patched: user role now allowed after tool role")
PYEOF

echo "Done. Mistral tool -> user role ordering relaxed."
