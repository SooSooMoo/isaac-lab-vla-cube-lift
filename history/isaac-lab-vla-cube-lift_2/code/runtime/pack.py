
import argparse
from pathlib import Path
import h5py
from smoke_utils import check_episode, state_equal, INSTRUCTIONS

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pick', required=True)
    p.add_argument('--above', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    output = Path(args.output)
    with h5py.File(args.pick, 'r') as pick, h5py.File(args.above, 'r') as above:
        sources = [pick, above]
        demos = [source['data/demo_0'] for source in sources]
        checks = [check_episode(demo) for demo in demos]
        if tuple(c[0] for c in checks) != INSTRUCTIONS:
            raise ValueError('Pass the pick episode to --pick and above episode to --above')
        if not state_equal(demos[0]['initial_state'], demos[1]['initial_state']):
            raise ValueError('Two commands must start from exactly the same saved state')
        for source in sources:
            if source['data'].attrs.get('observation_timing') != 'pre_action':
                raise ValueError('Unverified observation/action timing')
        output.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output, 'x') as dest:
            data = dest.create_group('data')
            for key, value in pick['data'].attrs.items():
                data.attrs[key] = value
            data.attrs['language_instruction'] = 'mixed_per_demo'
            data.attrs['total'] = sum(c[1] for c in checks)
            for i, (source, demo) in enumerate(zip(sources, demos)):
                source.copy(demo, data, name=f'demo_{i}')
    print('paired_dataset=OK', output, checks)

if __name__ == '__main__':
    main()
