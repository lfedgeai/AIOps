#!/usr/bin/env python3
"""Inspect desired EdgeLake table policies; safe dry-run by default.

This script deliberately does NOT guess the EdgeLake policy-registration API.
Use --register-command with a VERIFIED CLI template only after confirming
the syntax against the EdgeLake release in use. The template receives
{policy_json} (shell-quoted) and {policy_file} (shell-quoted).
"""
import argparse,json,pathlib,subprocess,shlex,urllib.request
ROOT=pathlib.Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--url',default='http://127.0.0.1:32159')
p.add_argument('--policy-file',default=str(ROOT/'config/table-policies.template.json'))
p.add_argument('--register-command',help='VERIFIED command template with {policy_json} or {policy_file}')
p.add_argument('--apply',action='store_true',help='requires --register-command')
a=p.parse_args()
entries=json.loads(pathlib.Path(a.policy_file).read_text())
assert isinstance(entries,list) and entries
for entry in entries:
 t=entry['table']; print(f"Policy: {t['dbms']}.{t['name']} ({len(t['create'])} DDL chars)")
try:
 req=urllib.request.Request(a.url,headers={'command':'blockchain get table','User-Agent':'curl/8.0.0'})
 with urllib.request.urlopen(req,timeout=8) as response: live=json.loads(response.read())
 existing={(e['table']['dbms'],e['table']['name']) for e in live if 'table' in e}
 print('Already registered:',', '.join(f'{db}.{tb}' for db,tb in sorted(existing)) or '(none)')
except Exception as exc:
 print('Could not query existing table policies:',exc); existing=set()
for entry in entries:
 t=entry['table']; key=(t['dbms'],t['name'])
 if key in existing:
  print('SKIP existing',*key); continue
 print('MISSING',*key)
 if not a.apply: continue
 if not a.register_command: p.error('--apply requires --register-command with syntax you verified')
 command=a.register_command.format(policy_json=shlex.quote(json.dumps(entry)),policy_file=shlex.quote(a.policy_file))
 print('Running verified registration command for',*key)
 subprocess.run(command,shell=True,check=True)
print('Completed inspection; no changes made' if not a.apply else 'Registration commands completed; verify tables before ingestion')
