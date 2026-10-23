"""Queue only CPU-heavy tools across model workers, retaining their exit status."""
import fcntl,os,subprocess,sys,time
args=sys.argv[1:]
if args and args[0]=='--':args=args[1:]
if not args:raise SystemExit('usage: heavy.py -- command args')
while True:
 for slot in range(3):
  f=open(f'/tmp/kwt-r6-heavy-{slot}.lock','a+')
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:f.close();continue
  env=os.environ.copy()
  env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
  print(f'HEAVY_SLOT={slot} argv={args!r}',flush=True)
  try:code=subprocess.call(args,env=env)
  finally:f.close()
  raise SystemExit(code if code>=0 else 128-code)
 time.sleep(.5)
