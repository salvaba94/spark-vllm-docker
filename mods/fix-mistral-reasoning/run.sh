#!/bin/bash
# Runtime mod: Fix Mistral reasoning_effort support in transformers tokenizer
#
# The transformers MistralCommonTokenizer.apply_chat_template rejects ALL
# kwargs, but vLLM passes reasoning_effort for tokenizer version >= 15.
# The underlying mistral_common library (>= 1.10.0) already supports
# reasoning_effort via ChatCompletionRequest and ModelSettingsBuilder,
# but the transformers glue layer blocks it.
#
# This mod patches transformers/tokenization_mistral_common.py to:
# 1. Pop reasoning_effort from kwargs before the rejection check
# 2. Pass it through to ChatCompletionRequest.from_openai()
#
# This enables [THINK]/[/THINK] reasoning mode for Mistral-Small-4 and
# similar v15+ Mistral models.
#
# Reference: https://github.com/vllm-project/vllm/pull/37081
#            https://github.com/huggingface/transformers/pull/41962

set -euo pipefail

TOKENIZER_FILE=$(python3 -c "
import transformers, os
print(os.path.join(os.path.dirname(transformers.__file__), 'tokenization_mistral_common.py'))
" 2>/dev/null)

if [ -z "$TOKENIZER_FILE" ] || [ ! -f "$TOKENIZER_FILE" ]; then
    echo "ERROR: Cannot find transformers/tokenization_mistral_common.py"
    exit 1
fi

echo "Patching: $TOKENIZER_FILE"

# Check if already patched
if grep -q 'reasoning_effort.*kwargs.pop' "$TOKENIZER_FILE"; then
    echo "  Already patched — skipping"
    exit 0
fi

# Patch 1: Extract reasoning_effort from kwargs before the rejection check,
# and pass it through to ChatCompletionRequest.from_openai()
python3 << 'PYEOF'
import re, sys

filepath = sys.argv[1] if len(sys.argv) > 1 else ""
if not filepath:
    # Get the path again
    import transformers, os
    filepath = os.path.join(os.path.dirname(transformers.__file__), 'tokenization_mistral_common.py')

with open(filepath, 'r') as f:
    content = f.read()

# Patch 1: Before "if kwargs: raise ValueError(...)", pop reasoning_effort
# Handle both transformers <5.3 (MistralCommonTokenizer) and >=5.3 (MistralCommonBackend)
import re
match = re.search(
    r'(        if kwargs:\n'
    r'            raise ValueError\(\n'
    r'                f"Kwargs \{list\(kwargs\.keys\(\)\)\} are not supported by `MistralCommon\w+\.apply_chat_template`\."\n'
    r'            \))',
    content
)

if not match:
    print("  WARNING: Could not find kwargs rejection block — may already be patched or changed")
    sys.exit(0)

old_kwargs_check = match.group(1)
# Preserve the original class name in the error message
new_kwargs_check = old_kwargs_check.replace(
    '        if kwargs:',
    '        # Pop reasoning_effort before kwargs rejection — it\'s supported by mistral_common\n'
    '        _reasoning_effort = kwargs.pop("reasoning_effort", None)\n'
    '        if kwargs:'
)

content = content.replace(old_kwargs_check, new_kwargs_check)

# Patch 2: Pass reasoning_effort to ChatCompletionRequest.from_openai()
old_from_openai = '''            chat_request = ChatCompletionRequest.from_openai(
                messages=messages,
                tools=tools,
                continue_final_message=continue_final_message,
            )'''

new_from_openai = '''            _from_openai_kwargs = {}
            if _reasoning_effort is not None:
                _from_openai_kwargs["reasoning_effort"] = _reasoning_effort
            chat_request = ChatCompletionRequest.from_openai(
                messages=messages,
                tools=tools,
                continue_final_message=continue_final_message,
                **_from_openai_kwargs,
            )'''

if old_from_openai not in content:
    print("  WARNING: Could not find ChatCompletionRequest.from_openai call — may be changed")
    sys.exit(0)

content = content.replace(old_from_openai, new_from_openai)

with open(filepath, 'w') as f:
    f.write(content)

print("  Patched: reasoning_effort now accepted and forwarded to mistral_common")
PYEOF

echo "Done. Mistral reasoning_effort support enabled."
