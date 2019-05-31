import pandas as pd
import numpy as np
# df=pd.DataFrame(data=[[1,2],[3,4]],columns=['a','b'])
# dd=df[df['a']>10]
# print(dd.empty)
# if dd.empty:
#     print("dd is empty")

myarray=[[1,2,3]]

# print(np.isnan(myarray).any())
dd=pd.DataFrame(myarray)

print(pd.isnull(pd.DataFrame(myarray)))

if  in pd.isnull(pd.DataFrame(myarray)).any():
    print(True)
# if  pd.isnull(dd).any() :
#     print('dd', dd)
