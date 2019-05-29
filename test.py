from data_aug.data_aug import *
from data_aug.bbox_util import *
import cv2 
import pickle as pkl
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import matplotlib.image as mp

path_axis=r'F:\陶士来文件\tsl_python_project\object_detection\data_examples\train_labels_new.csv'
df=pd.read_csv(path_axis)
print(df.shape,df.head())
# # dd = df[(df['class'] == 'alibaba') & (df['filename'] == '1.jpg')][['xmin', 'ymin', 'xmax', 'ymax']]
# #感觉有点像join的连接方式，造成很多重复的行，例如
# dd=df.loc[(df['class'] == 'alibaba') & (df['filename'] == '1.jpg'),['xmin', 'ymin', 'xmax', 'ymax']].drop_duplicates()
# print('dd_shape={},dd={}'.format(dd.shape, dd))



# df.groupby(['class','filename']).apply(lambda ddf :print(ddf,'\n'))
# dd=df[(df['class']=='alibaba') & (df['filename']=='1.jpg')][['xmin','ymin','xmax'  ,'ymax'    ]]
# print(dd.values.tolist())


# img = cv2.imread("messi.jpg")[:,:,::-1] #OpenCV uses BGR channels
# bboxes = pkl.load(open("messi_ann.pkl", "rb"))
#
# print(bboxes)
#
# transforms = Sequence([RandomHorizontalFlip(1), RandomScale(0.2, diff = True), RandomRotate(10)])
#
# img, bboxes = transforms(img, bboxes)
# print(bboxes)

# plt.imshow(draw_rect(img, bboxes))
# plt.show()

def data_aug():
    rootpath_jpg=r'E:\tsl_file\windows_v1.8.1\windows_v1.8.1\label_data_self\raw'
    g = os.walk(rootpath_jpg)
    index=0
    df_all=pd.DataFrame()
    for path,dir_list,file_list in g:
        # if index==2:
        #     break
        for file_name in file_list:
            try :
                # print(os.path.join(path, file_name) )
                class_name=path.split('\\')[-1]
                dd = df[(df['class'] == class_name) & (df['filename'] == file_name)][['xmin', 'ymin', 'xmax', 'ymax']]
                # print('dd_shape={},dd={}'.format(dd.shape,dd))
                print(path,file_name)
                bboxes=np.array(dd.values,dtype=np.float64)

                # print('filename={},class={}'.format(file_name, class_name))
                # print('bboxes_shape={},bboxes={}'.format(bboxes.shape, bboxes))
                image_path=os.path.join(path,file_name)
                print('image_path',image_path)
                img=cv2.imread(image_path)[:,:,::-1] #OpenCV uses BGR channels
                # print('img',img)
                # img = Image.open(image_path)

                transforms = Sequence([RandomHorizontalFlip(1), RandomScale(0.2, diff = True), RandomRotate(10)])
                # plt.imshow(img)
                # plt.show()
                img, bboxes = transforms(img, bboxes)
                print('filename2={},class2={}'.format(file_name,class_name))
                # print('bboxes2_shape={},bboxes2={}'.format(bboxes.shape,bboxes))
                # aug_img=draw_rect(img, bboxes)
                # plt.imshow(aug_img)
                # plt.show()

                'filename,width,height,class,xmin,ymin,xmax,ymax'

                df_temp=pd.DataFrame(data=bboxes,columns=['xmin', 'ymin', 'xmax', 'ymax'])
                df_temp['filename']=file_name
                df_temp['class']=class_name
                print('df_temp',df_temp)
                df_all=df_all.append(df_temp)


                if not os.path.exists('raw_augment/{}'.format(class_name)):
                    os.makedirs('raw_augment/{}'.format(class_name))
                    mp.imsave('raw_augment/{}/{}'.format(class_name, file_name), img)
                else:
                    mp.imsave('raw_augment/{}/{}'.format(class_name,file_name), img)
            except Exception as e:
                print('my err',e)




        # index=index+1
    df_all.to_csv('train_labels_new.csv',index=False)
    print(df_all)


if __name__=='__main__':
    data_aug()

