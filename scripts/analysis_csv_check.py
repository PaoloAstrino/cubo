import csv, os, math
os.chdir('c:/Users/paolo/Desktop/cubo')
path='results/tonight_full/analysis_full.csv'
print(os.path.exists(path), os.path.getsize(path)/1024)
rows=list(csv.DictReader(open(path,'r',encoding='utf-8')))
print('Total rows:', len(rows))

def is_nan_field(val):
    if val in ['', 'None', 'nan', 'NaN', 'NULL', None]:
        return True
    try:
        v=float(val)
        return math.isnan(v)
    except Exception:
        return True

count_nan_faith=sum(1 for r in rows if is_nan_field(r['faithfulness']))
count_nan_relev=sum(1 for r in rows if is_nan_field(r['relevancy']))
count_nan_precision=sum(1 for r in rows if is_nan_field(r['precision']))
print('NaN counts in csv: faith:', count_nan_faith, 'relev:', count_nan_relev, 'prec:', count_nan_precision)

print('\nSample queries with NaN faith (first 10):')
ct=0
for r in rows:
    if is_nan_field(r['faithfulness']):
        print(r['index'], r['query'][:120], 'relev:', r['relevancy'])
        ct+=1
        if ct>=10:
            break

print('\nSample queries with computed faith (first 10):')
ct=0
for r in rows:
    if not is_nan_field(r['faithfulness']):
        print(r['index'], r['query'][:120], 'faith:', r['faithfulness'], 'relev:', r['relevancy'])
        ct+=1
        if ct>=10:
            break
