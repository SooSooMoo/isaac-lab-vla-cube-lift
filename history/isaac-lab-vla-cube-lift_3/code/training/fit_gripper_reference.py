"""Solve frozen-GRU gripper constraints; validate sequential float32 before rollout."""
from pathlib import Path
import os, sys, json, hashlib, tempfile, shutil, subprocess

ROOT = Path('/workspace/step4')
SOURCE = ROOT / 'vla_northwest_gripper780_fit_lmmqdnik'
CACHE = SOURCE / 'recovery_features.pt'
AUDIT = ROOT / 'vla_earlyclose_action_audit_dh7dux8_'
PRESERVE = ROOT / 'vla_axis_preserve5_complete_3do4dmss/manifest.json'
TEACHER = ROOT / 'vla_new21_earlyclose_recovery_inb4kads/manifest.json'
MARGIN = 1e-4

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def main():
    import numpy as np
    import torch
    from scipy.optimize import linprog
    from scipy import sparse
    from temporal_action import TemporalAction

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    assert torch.cuda.is_available(), 'CUDA unavailable'
    bundle = torch.load(SOURCE/'candidate.pt', map_location='cpu', weights_only=True)
    assert bundle['format'] == 'pick_temporal_action_v1'
    assert bundle['instruction'] == 'pick up the cube'
    assert sha(SOURCE/'temporal_action.py') == bundle['temporal_code_sha256']
    source_hash=sha(SOURCE/'candidate.pt')
    assert source_hash=='84a8efdd7c6feb9fee37e66132ca7c5efcb0398680cc8c7f4c73f173ccb0f7e2'
    evidence_file=ROOT/'vla_fixed29_validation_vdbnuo_d/combined29.json'
    evidence=json.loads(evidence_file.read_text())
    assert len(evidence)==29 and sum(r['pass'] for r in evidence)==26
    rows=[]
    for item in evidence:
        if not item['pass']:
            continue
        folder=Path(item['result_directory'])
        summary=json.loads((folder/'summary.json').read_text())
        assert summary['checkpoint_sha256']==source_hash
        assert item['learned_policy'] and not item['runtime_action_override']
        data_file=folder/'preserve_features.pt'
        if not data_file.is_file():
            raise RuntimeError('Missing saved success features: '+str(data_file))
        result=json.loads((folder/'direct_result.json').read_text())
        assert sha(data_file)==result['preserve_features_sha256']
        d=torch.load(data_file,map_location='cpu',weights_only=True)
        assert d['parent_sha256']==bundle['parent_sha256'] and d['instruction']=='pick up the cube'
        assert d['source_checkpoint_sha256']==source_hash
        with np.load(folder/'teacher_trace.npz',allow_pickle=False) as tr:
            assert np.array_equal(d['target'].numpy(),tr['actions'])
        assert torch.equal(d['source_logit'].reshape(-1)>=0,d['target'][:,6]<0)
        rows.append(dict(features=d['features'][:,:256],target=d['target'].clone(),offset=d['offset_mm'],repair=False))
    diagnostic_file=ROOT/'vla_interior_close3_diagnostic_ma4n0jrj/results.json'
    diagnostic_rows=json.loads(diagnostic_file.read_text())
    assert len(diagnostic_rows)==3
    schedules={(0,10):618,(-7,12):694,(-7,-12):741}
    for rr in diagnostic_rows:
        assert rr['pass'] and rr['diagnostic_only'] and rr['runtime_action_override']
        assert rr['diagnostic_comparison_valid'] and all(rr['prefix_matches_reference'].values())
        folder=Path(rr['result_directory'])
        summary=json.loads((folder/'summary.json').read_text())
        assert summary['checkpoint_sha256']==source_hash
        d=torch.load(folder/'diagnostic_features.pt',map_location='cpu',weights_only=True)
        assert d['instruction']=='pick up the cube' and d['diagnostic_only'] and d['runtime_action_override']
        target=d['executed_actions'].float()
        cut=schedules[tuple(rr['offset_mm'])]
        assert (target[:cut,6]>0).all() and (target[cut:,6]<0).all()
        assert torch.equal(d['predicted_actions'][:,:6],target[:,:6])
        with np.load(folder/'teacher_trace.npz',allow_pickle=False) as tr:
            assert np.array_equal(target.numpy(),tr['actions'])
        rows.append(dict(features=d['features'],target=target,offset=rr['offset_mm'],repair=True))
    assert len(rows)==29 and len({tuple(r['offset']) for r in rows})==29
    for r in rows:
        assert r['features'].shape == (len(r['target']),256)
        assert r['target'].shape[1] == 7
        assert torch.isfinite(r['features']).all() and torch.isfinite(r['target']).all()
    out = Path(tempfile.mkdtemp(prefix='vla_interior_gripper29_fit_',dir=ROOT))
    print('RUN_DIRECTORY',out,flush=True)
    (out/'protocol.json').write_text(json.dumps(dict(source=str(SOURCE/'candidate.pt'),source_sha256=sha(SOURCE/'candidate.pt'),preservation_results_sha256=sha(evidence_file),diagnostic_features_sha256=sha(diagnostic_file),scope='preserve 26 successful trajectory labels and fit three successful diagnostic closure sequences; saved-state constraints are not rollout proof',runtime_action_override=False),indent=2))
    model = TemporalAction().cuda().eval()
    model.load_state_dict(bundle['temporal_state'],strict=True)
    captured = {}
    hook = model.output.register_forward_pre_hook(lambda m,args: captured.update(hidden=args[0].detach()))
    matrices=[]; labels=[]; old_arms=[]; base_errors=[]
    with torch.inference_mode():
        for r in rows:
            h=None; hidden=[]; arms=[]; logits=[]
            for f in r['features'].cuda():
                a,g,h=model(f.reshape(1,1,256),h)
                hidden.append(captured['hidden'].reshape(128).cpu())
                arms.append(a.reshape(6).cpu()); logits.append(g.reshape(()).cpu())
            matrix=torch.stack(hidden).numpy().astype(np.float64)
            matrices.append(np.column_stack([matrix,np.ones(len(matrix))]))
            # Preserve this checkpoint's decisions on saved states; change only the demonstrated startup error.
            closed=torch.stack(logits)>=0
            if not r['repair']:
                assert torch.equal(closed,r['target'][:,6]<0), ('Successful trajectory replay mismatch',r['offset'])
            if r['repair']:
                print('CLOSE_ONSET_BEFORE',r['offset'],float(torch.stack(logits)[int((r['target'][:,6]<0).nonzero()[0].item())]),flush=True)
                closed=r['target'][:,6]<0
            r['target'][:,6]=torch.where(closed,-1.,1.)
            labels.append(np.where(closed.numpy(),1.,-1.))
            old_arms.append(torch.stack(arms))
            base_errors.append(int(((torch.stack(logits)>=0)!=(r['target'][:,6]<0)).sum()))
    hook.remove()
    print('SCOPE current saved-state decisions preserved with full new teacher sequence constraints; not autonomous preservation proof',flush=True)
    A=np.vstack(matrices); y=np.concatenate(labels)
    old=np.r_[bundle['temporal_state']['output.weight'][6].numpy(),bundle['temporal_state']['output.bias'][6].item()].astype(np.float64)
    assert A.shape[1] == len(old) == 129
    # delta and nonnegative t, minimize sum(t), with -t <= delta <= t.
    def solve(indices,scope):
        signed=y[indices,None]*A[indices]
        n=len(old); eye=sparse.eye(n,format='csr'); zero=sparse.csr_matrix((len(indices),n))
        lhs=sparse.vstack([sparse.hstack([-sparse.csr_matrix(signed),zero]),sparse.hstack([eye,-eye]),sparse.hstack([-eye,-eye])],format='csr')
        rhs=np.r_[signed@old-MARGIN,np.zeros(2*n)]
        sol=linprog(np.r_[np.zeros(n),np.ones(n)],A_ub=lhs,b_ub=rhs,bounds=[(None,None)]*n+[(0,None)]*n,method='highs',options={'time_limit':120,'primal_feasibility_tolerance':1e-8,'dual_feasibility_tolerance':1e-8})
        report=dict(scope=scope,status=int(sol.status),message=sol.message,samples=len(indices),required_margin=MARGIN)
        weights=None
        if sol.success:
            weights=old+sol.x[:n]
            margin=y[indices]*(A[indices]@weights)
            report.update(achieved_min_margin=float(margin.min()),double_sign_errors=int((margin<=0).sum()),parameter_change_l1=float(np.abs(weights-old).sum()),max_abs_parameter=float(np.abs(weights).max()))
            assert margin.min() >= MARGIN-1e-7, 'Solver constraint residual too large'
        print('FEASIBILITY_RESULT',json.dumps(report),flush=True)
        return weights,report
    weights,report=solve(np.arange(len(y)),'all_preservation_and_full_teacher_sequence')
    reports=[report]
    if weights is None:
        (out/'feasibility.json').write_text(json.dumps(reports,indent=2))
        print('EVALUATION_SKIPPED no full-constraint solution; status 2 means infeasible at stated margin, other statuses are inconclusive',flush=True)
        return 0
    state={k:v.clone() for k,v in bundle['temporal_state'].items()}
    state['output.weight'][6]=torch.tensor(weights[:128],dtype=state['output.weight'].dtype)
    state['output.bias'][6]=float(weights[128])
    for k,v in bundle['temporal_state'].items():
        assert torch.equal(state[k][:6],v[:6]) if k in ('output.weight','output.bias') else torch.equal(state[k],v),k
    model.load_state_dict(state,strict=True)
    errors=[]; max_arm=0.
    with torch.inference_mode():
        for index,r in enumerate(rows):
            h=None; values=[]; arms=[]
            for f in r['features'].cuda():
                a,g,h=model(f.reshape(1,1,256),h)
                values.append(g.reshape(()).cpu());arms.append(a.reshape(6).cpu())
            g=torch.stack(values); truth=r['target'][:,6]<0
            max_arm=max(max_arm,float((torch.stack(arms)-old_arms[index]).abs().max()))
            errors.append(dict(offset=r['offset'],false_close=int(((g>=0)&~truth).sum()),missed_close=int(((g<0)&truth).sum()),min_signed_logit=float((g*torch.where(truth,1.,-1.)).min())))
    gate=all(r['false_close']==0 and r['missed_close']==0 and r['min_signed_logit']>0 for r in errors) and max_arm==0
    summary=dict(float32_gate=gate,shared_gru_arm_unchanged=True,same_input_arm_max_difference=max_arm,episodes=errors,solver=reports,rollout_verified=False)
    (out/'feasibility.json').write_text(json.dumps(summary,indent=2))
    print('FLOAT32_GATE',json.dumps(summary),flush=True)
    if not gate:
        print('EVALUATION_SKIPPED float32 verification failed; mathematical solution not accepted',flush=True)
        return 0
    candidate=dict(bundle);candidate.update(temporal_state=state,rollout_verified=False,gripper_solver='minimum_L1_change_linear_constraints')
    torch.save(candidate,out/'candidate.pt')
    for name in ('temporal_action.py','pick_model.py','pick_inputs.py','features.pt'):
        (out/name).symlink_to((SOURCE/name).resolve())
    # Keep the evaluator implementation unchanged; substitute the position list only.
    import ast
    base=(SOURCE/'repair1.py').read_text()
    del model,captured
    torch.cuda.empty_cache()
    all_results=[]
    for stage,positions in [('repair3',[(0,10),(-7,12),(-7,-12)])]:
        nodes=[n for n in ast.walk(ast.parse(base)) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='positions' for t in n.targets)]
        assert len(nodes)==1,'Unexpected evaluator position declaration'
        node=nodes[0];lines=base.splitlines(True)
        lines[node.lineno-1:node.end_lineno]=[' '*node.col_offset+'positions = '+repr(positions)+'\n']
        code=''.join(lines);compile(code,stage,'exec')
        evaluator=out/(stage+'.py');evaluator.write_text(code)
        exitcode=subprocess.run([sys.executable,'-u',str(evaluator),'--checkpoint',str(out/'candidate.pt'),'--output-dir',str(out/stage)]).returncode
        if exitcode:
            print('EVALUATION_INCOMPLETE',out/stage,flush=True);return exitcode
        result_path=out/stage/'results.json'
        if not result_path.is_file():
            print('EVALUATION_INCOMPLETE',out/stage,flush=True);return 1
        results=json.loads(result_path.read_text())
        assert len(results)==len(positions)
        assert all(r['learned_policy'] and r['runtime_action_override'] is False and r['language_encoder_calls']==r['model_action_calls']==r['samples'] for r in results)
        all_results.extend(dict(r,evaluation_stage=stage) for r in results)
        (out/'combined_results.json').write_text(json.dumps(all_results,indent=2))
        if not all(r['pass'] for r in results):
            print('PRESERVATION_EVALUATION_SKIPPED repair3 not all passed',flush=True);break
    print('VLA_INTERIOR_GRIPPER29_SUMMARY',flush=True)
    print('dx_mm dy_mm | pick | hold3s | pass | hold_max_xy_mm',flush=True)
    for r in all_results:
        xy=r.get('hold_maximum_xy_mm')
        print(*r['offset_mm'],'|',r['pick_pass'],'|',r['hold_3s_pass'],'|',r['pass'],'|',round(xy,2) if xy is not None else '-',flush=True)
    final=dict(passed=sum(r['pass'] for r in all_results),trials=len(all_results),required_trials=3,checkpoint=str(out/'candidate.pt'),results=str(out/'combined_results.json'))
    (out/'status.json').write_text(json.dumps(final,indent=2))
    print('INTERIOR_GRIPPER29_COMPLETE',json.dumps(final),flush=True)
    return 0

if __name__=='__main__':
    if '--prepared' not in sys.argv:
        env=os.environ.copy()
        env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        env['PYTHONPATH']=os.pathsep.join([str(SOURCE),str(ROOT/'.runtime_packages'),str(ROOT/'python_packages'),env.get('PYTHONPATH','')])
        env['PYTHONDONTWRITEBYTECODE']='1'
        os.execve(sys.executable,[sys.executable,'-u',__file__,'--prepared'],env)
    try:
        raise SystemExit(main())
    except Exception as exc:
        print('FEASIBILITY_STOP',type(exc).__name__,str(exc),flush=True)
        raise SystemExit(1)
