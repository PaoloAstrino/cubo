import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
sys.path.insert(0, os.getcwd())
os.chdir('c:/Users/paolo/Desktop/cubo')
from cubo.core import CuboCore
from cubo.config import config
config.set('vector_store_path','results/tonight_full/storage')

print('Initializing CuboCore and components...')
core=CuboCore()
core.initialize_components()
print('Components initialized, querying...')
res=core.query_retrieve('Where should I park my rainy-day / emergency fund?', top_k=10)
print('Retrieved', len(res), 'items')
for i,item in enumerate(res):
    print('\n--- item', i)
    print('id', item.get('id'))
    print('metadata id', (item.get('metadata') or {}).get('id'))
    print('similarity', item.get('similarity'))
    doc = item.get('document') or item.get('text') or item.get('content')
    print('doc start:', (doc[:160] + '...') if doc else 'None')

print('Done')
