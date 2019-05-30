import pandas as pd
df=pd.DataFrame(data=[[1,2],[3,4]],columns=['a','b'])
dd=df[df['a']>10]
print(dd.empty)
if dd.empty:
    print("dd is empty")