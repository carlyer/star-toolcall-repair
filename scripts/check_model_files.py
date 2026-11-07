"""Check local weight shards exist before any GPU service starts."""
import json
import sys
from pathlib import Path

def check_model(path):
    root = Path(path)
    if not (root/'config.json').is_file():
        return False, 'missing config.json'
    indices = list(root.glob('*.index.json'))
    shards = set()
    for index in indices:
        mapping = json.loads(index.read_text()).get('weight_map', {})
        shards.update(mapping.values())
    if not shards:
        shards = {p.name for pattern in ('*.safetensors', 'pytorch_model*.bin') for p in root.glob(pattern)}
    if not shards:
        return False, 'missing model weights'
    missing = [name for name in shards if not (root/name).is_file() or (root/name).stat().st_size == 0]
    return not missing, f'missing shards: {missing}' if missing else f'{len(shards)} weight shards found'

if __name__ == '__main__':
    for path in sys.argv[1:]:
        valid, reason = check_model(path)
        print(f'{path}: {reason}')
        if not valid:
            raise SystemExit(1)
