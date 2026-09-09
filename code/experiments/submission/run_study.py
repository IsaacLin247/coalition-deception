#!/usr/bin/env python3
"""Run the complete post-audit replication without mixing archived or partial results.

Create/freeze a protocol before launch, then run disjoint workers against the same
source snapshot. A success marker requires exit=0 AND every required artifact.
There is no early stopping or selection of seeds based on outcomes.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))
from social_collusion.runmeta import source_manifest, source_digest


def study_jobs():
    jobs = []
    def add(name, script, args, artifacts, family, worker='desktop', dependencies=()):
        options=dict(zip(map(str,args[::2]),map(str,args[1::2])))
        budget=int(options.get('--updates',options.get('--updates-per-stage',400)))
        stages={}
        interval=50
        if family=='f1':
            stages={'':budget}; interval=20
        elif family in ('f3','f4','budget'):
            stages={'coalition0_vs_soft_credibility':int(options['--initial-updates']),
                    'crew1_vs_learned_coalition0':int(options['--defender-updates']),
                    'coalition1_vs_learned_crew1':int(options['--adapted-updates'])}
        elif family in ('hypothesis','counterattack'):
            stages={f'coalition_vs_{d}':budget for d in options['--defenses'].split(',')}
        elif family=='ablation':
            stages={f'coalition_{d}':budget for d in ('default','no_channel','no_partner','no_channel_no_partner')}
        elif family in ('multigen','dependence','reward'):
            stages={'stage00_C0':budget}
            for g in range(1,int(options['--generations'])+1):
                stages[f'stage{2*g-1:02d}_D{g}']=budget
                stages[f'stage{2*g:02d}_C{g}']=budget
            artifacts=[*artifacts,'evaluation_records.json']
        jobs.append(dict(name=name, family=family, worker=worker,
                         command=[script, *map(str,args), '--name',name],
                         artifacts=list(artifacts), dependencies=list(dependencies),
                         expected_stages=stages, checkpoint_interval=interval))
    common = ['--n-envs',32,'--rollout-episodes',32,'--eval-every',50,
              '--eval-episodes',200,'--crossplay-episodes',1000,
              '--checkpoint-every',50,'--replay-episodes',3,'--threads',1,'--device','cpu']
    mg = ['--device','cpu','--threads',1]
    # Place a complete 5+2 replication first; ordering never changes its budget.
    for crew in (5,3,7):
        for seed in range(10):
            add(f'f1_tenseed_crew{crew}_s{seed}', 'scripts/train_emergence.py',
                ['--rule','soft_credibility','--n-agents',crew+2,'--updates',400,
                 '--eval-every',20,'--eval-episodes',400,'--checkpoint-every',20,
                 '--n-envs',64,'--rollout-episodes',64,'--seed',seed,'--threads',1],
                ['result.json','history.json','checkpoint_0.pt','checkpoint_400.pt','checkpoint_final.pt'], 'f1','local')
    for rounds, fam in ((1,'f3'),(8,'f4')):
        for crew in (5,3,7):
            for seed in range(10):
                add(f'{fam}_tenseed_crew{crew}_s{seed}', 'scripts/train_f3_matched_cycle.py',
                    ['--n-agents',crew+2,'--max-rounds',rounds,'--seed',seed,
                     '--initial-updates',400,'--defender-updates',400,'--adapted-updates',400,
                     '--single-round-objective','balanced_ejection',*common],
                    ['result.json','crossplay_summary.csv','episode_metrics.csv','round_metrics.csv'],fam)
    for crew in (5,3,7):
        for seed in range(10):
            add(f'f2_tenseed_crew{crew}_s{seed}','scripts/evaluate_f2_seed.py',
                ['--crew',crew,'--seed',seed,'--episodes',1000,'--threads',1],
                ['f2_seed_results.json','f2_seed_results.csv'],'f2','local')
    for n, seeds, defs, tag in ((7,10,'soft,mean,hypothesis','hyp_'),(9,5,'soft,mean,hypothesis','hyp_'),
                                (7,10,'soft,rule,both',''),(5,5,'soft,rule,both','')):
        for seed in range(seeds):
            add(f'counterattack_{tag}n{n}_r1_s{seed}','experiments/dependence/train_counterattack.py',
                ['--n-agents',n,'--seed',seed,'--defenses',defs,*mg],
                ['result.json','crossplay.csv'], 'hypothesis' if tag else 'counterattack')
    for rounds, budgets in ((1,(800,1600)),(8,(800,))):
        for budget in budgets:
            for seed in range(5):
                add(f'f{3 if rounds==1 else 4}_budget_d{budget}_crew5_s{seed}',
                    'scripts/train_f3_matched_cycle.py',
                    ['--n-agents',7,'--max-rounds',rounds,'--seed',seed,'--initial-updates',400,
                     '--defender-updates',budget,'--adapted-updates',400,
                     '--single-round-objective','balanced_ejection',*common],
                    ['result.json','crossplay_summary.csv','episode_metrics.csv','round_metrics.csv'],'budget')
    for rounds in (1,8):
        for seed in range(5):
            add(f'ablation_n7_r{rounds}_s{seed}','experiments/ablation/train_coalition_ablation.py',
                ['--n-agents',7,'--max-rounds',rounds,'--seed',seed,*mg],
                ['result.json','result.csv'],'ablation')
    for n,r,k,seeds,defense in ((7,8,10,10,'none'),(7,1,10,10,'none'),(5,8,10,10,'none'),
                               (9,8,3,5,'none'),(7,8,4,10,'dependence')):
        for seed in range(seeds):
            add(f'multigen_{defense}_n{n}_r{r}_k{k}_s{seed}', 'experiments/multigen/run_multigen.py',
                ['--n-agents',n,'--max-rounds',r,'--generations',k,'--defense',defense,'--seed',seed,*mg],
                ['crossplay.json','policy_matrices.json','generations.json'],'multigen' if defense=='none' else 'dependence')
    for tag,obj,rounds,k in (('bal','balanced_ejection',1,4),('meet','meeting_ejection',1,4),('fe','false_ejection',8,2)):
        for seed in range(5):
            reward_args = ['--single-round-objectives',f'{obj},{obj}'] if rounds==1 else ['--multi-round-objective',obj]
            add(f'multigen_rw{tag}_n7_r{rounds}_k{k}_s{seed}','experiments/multigen/run_multigen.py',
                ['--n-agents',7,'--max-rounds',rounds,'--generations',k,'--defense','none',
                 '--seed',seed,*reward_args,*mg],
                ['crossplay.json','policy_matrices.json','generations.json'],'reward')
    for seed in range(5):
        dep=f'multigen_none_n7_r1_k10_s{seed}'
        add(f'depstatic_n7_r1_s{seed}','experiments/dependence/evaluate_static.py',
            ['--n-agents',7,'--seed',seed,'--checkpoints',f'{{results}}/{dep}',
             '--max-learned-generation',3,'--threads',1],
            ['result.json','rules.csv','strength_sweep.csv','learned.csv'],'static',dependencies=[dep])
    return jobs


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
    temp.replace(path)


def job_digest(job):
    return hashlib.sha256(json.dumps({k:v for k,v in job.items() if k!='worker'},sort_keys=True).encode()).hexdigest()


def valid_artifacts(job, results, deep=False):
    for name in job['artifacts']:
        p=results/job['name']/name
        if not p.is_file() or p.stat().st_size==0:
            return False
        if p.suffix=='.json':
            try: json.loads(p.read_text())
            except (ValueError,OSError): return False
    for stage,budget in job.get('expected_stages',{}).items():
        history=results/job['name']/stage/'history.json'
        if not (history.parent/'checkpoint_final.pt').is_file():
            return False
        try:
            payload=json.loads(history.read_text())
            if (not payload.get('final_metrics') or payload.get('updates_completed')!=budget
                or len(payload.get('history',[]))!=budget):
                return False
        except (ValueError,OSError): return False
        for update in range(job['checkpoint_interval'],budget+1,job['checkpoint_interval']):
            if not (history.parent/f'checkpoint_{update}.pt').is_file():
                return False
    index=results/job['name']/'evaluation_records.json'
    if deep and index.exists():
        try:
            records=json.loads(index.read_text())['evaluations']
            generations=int(dict(zip(job['command'][1::2],job['command'][2::2]))['--generations'])
            if len(records)!=(2*generations+1)+(generations+1)**2:
                return False
            for record in records:
                for key,total in (('episode_metrics','n_episodes'),('round_metrics','n_meetings')):
                    path=results/job['name']/record[key].replace('\\','/')
                    with path.open(newline='') as fh:
                        if sum(1 for _ in csv.DictReader(fh))!=record[total]:
                            return False
        except (ValueError,OSError,KeyError,TypeError): return False
    return True


def completed(job, results, fingerprint):
    marker=results/'_control'/f"{job['name']}.complete.json"
    if not marker.exists(): return False
    try: row=json.loads(marker.read_text())
    except (ValueError,OSError): return False
    return (row.get('source_sha256')==fingerprint and row.get('job_sha256')==job_digest(job)
            and row.get('returncode')==0 and valid_artifacts(job,results))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['freeze','list','run','status'])
    ap.add_argument('--protocol',type=Path,default=REPO/'submission_protocol.json')
    ap.add_argument('--results',type=Path,default=REPO/'results'/'submission_20260909')
    ap.add_argument('--worker',choices=['local','desktop','all'],default='all')
    ap.add_argument('--slots',type=int,default=4)
    ap.add_argument('--families',default='')
    args=ap.parse_args()
    if args.action=='freeze':
        if args.protocol.exists(): raise SystemExit('Refusing to overwrite a frozen protocol.')
        files=source_manifest()
        write_json(args.protocol,dict(study='post_audit_replication_20260909',
            created_utc=dt.datetime.now(dt.timezone.utc).isoformat(), source_sha256=source_digest(files),
            sources=files, jobs=study_jobs(),
            protocol=dict(training_seed_counts='10 core / 5 supplementary; all planned seeds retained',
                training_updates='400 per stage except prespecified defender budgets 800/1600',
                inference='Seed-level paired effects; 95% bootstrap intervals; exact paired mean sign flips; Holm correction within declared families',
                selection='No outcome-dependent stopping, seed exclusion, hyperparameter tuning or checkpoint selection',
                evaluation='Fixed episode-index batches; independent final evidence base 1987654321; monitored curves base 987654321',
                observation='Full within-round private history for new spatial actors; legacy checkpoints retain compressed schema',
                status='Post-audit replication protocol; not a prospective preregistration of the original study')))
        print(f'Frozen {len(study_jobs())} jobs -> {args.protocol}'); return 0
    data=json.loads(args.protocol.read_text()); fingerprint=data['source_sha256']
    jobs=[j for j in data['jobs'] if (args.worker=='all' or j['worker']==args.worker)
          and (not args.families or j['family'] in args.families.split(','))]
    results=args.results.resolve(); controls=results/'_control'; logs=results/'_logs'
    pending=[j for j in jobs if not completed(j,results,fingerprint)]
    if args.action in ('list','status'):
        print(json.dumps(dict(total=len(jobs),complete=len(jobs)-len(pending),pending=len(pending),
                              source_sha256=fingerprint),indent=2))
        if args.action=='list':
            for j in pending: print(j['worker'],j['name'])
        return 0
    if args.slots<1: raise SystemExit('slots must be positive')
    if source_digest()!=fingerprint: raise SystemExit('Source differs from frozen protocol; launch its snapshot.')
    controls.mkdir(parents=True,exist_ok=True); logs.mkdir(parents=True,exist_ok=True)
    lock=controls/f'worker_{args.worker}.lock'
    try:
        fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(f'Worker lock exists: {lock}; inspect the recorded process before restarting.')
    os.write(fd,json.dumps(dict(pid=os.getpid(),started_utc=dt.datetime.now(dt.timezone.utc).isoformat())).encode()); os.close(fd)
    env=dict(os.environ,SOCIAL_COLLUSION_RUNS=str(results),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
             OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',PYTHONPATH=str(REPO/'src'))
    running={}; failed=[]; all_jobs={j['name']:j for j in data['jobs']}
    def status():
        write_json(controls/f'worker_{args.worker}.json',dict(updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            source_sha256=fingerprint,total=len(jobs),completed=sum(completed(j,results,fingerprint) for j in jobs),
            running=[dict(name=n,pid=v[0].pid) for n,v in running.items()], pending=[j['name'] for j in pending],failed=failed))
    try:
        while pending or running:
            for job in pending[:]:
                if len(running)>=args.slots: break
                if any(not completed(all_jobs[d],results,fingerprint) for d in job['dependencies']): continue
                if source_digest()!=fingerprint: raise RuntimeError('Source changed during queue; refusing new jobs.')
                pending.remove(job)
                cmd=[sys.executable,'-u',*[s.replace('{results}',str(results)) for s in job['command']]]
                log=open(logs/f"{job['name']}.log",'a',buffering=1)
                log.write('\n'+json.dumps(dict(started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),command=cmd))+'\n')
                p=subprocess.Popen(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
                running[job['name']]=(p,time.monotonic(),job,log)
                print('START',job['name'], 'pid',p.pid,flush=True)
            for name,(p,start,job,log) in list(running.items()):
                rc=p.poll()
                if rc is None: continue
                log.close(); del running[name]
                record=dict(name=name,returncode=rc,wall_s=round(time.monotonic()-start,2),
                    finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),source_sha256=fingerprint,
                    job_sha256=job_digest(job),
                    artifacts_valid=valid_artifacts(job,results,deep=True))
                success=rc==0 and record['artifacts_valid']
                write_json(controls/f"{name}.{'complete' if success else 'failed'}.json",record)
                if not success: failed.append(record)
                print('DONE' if success else 'FAILED',name,record['wall_s'],flush=True)
            status()
            if pending and not running and all(any(not completed(all_jobs[d],results,fingerprint) for d in j['dependencies']) for j in pending):
                print('Remaining jobs have incomplete dependencies.',flush=True); break
            time.sleep(5)
        status()
    finally:
        # On interruption terminate children instead of leaving untracked competing jobs.
        for p,_,_,log in running.values():
            p.terminate(); log.close()
        lock.unlink(missing_ok=True)
    return 1 if failed or pending else 0

if __name__=='__main__':
    raise SystemExit(main())
