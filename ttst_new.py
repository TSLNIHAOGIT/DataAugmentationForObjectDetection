from data_aug.data_aug import *
from data_aug.bbox_util import *
import cv2 
import pickle as pkl
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import matplotlib.image as mp
'''
将每一个文件下图片的所有变换仍然存在该图片的文件夹下，因此每一次变换的文件名称要不同：该文件夹下，变换名称+次数

'''
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



def each_jpg_transform(transforms_name,df_all,img,bboxes,file_name,folder_name,dd,times):
    '''单个的用每个名称，多个的用sequence_m1 和sequence_m2'''

    if transforms_name=='sequence1_model':
        transforms_model = [RandomHorizontalFlip(0.1), RandomScale(0.3, diff=True), RandomRotate(25),
                            RandomTranslate(0.2, diff=True), RandomShear(0.15)]
    elif transforms_name=='sequence2_model':
        transforms_model = [RandomHorizontalFlip(0.1), RandomScale(0.3, diff=True), RandomRotate(25),
                            RandomTranslate(0.2, diff=True), RandomShear(0.15),RandomHSV(100, 100, 100)]
    elif transforms_name=='RandomHorizontalFlip':
        transforms_model = [RandomHorizontalFlip(0.8)]
    elif transforms_name=='RandomScale':
        transforms_model = [ RandomScale(0.3, diff=True)]
    elif transforms_name=='RandomTranslate':
        transforms_model = [RandomTranslate(0.2, diff = True)]
    elif transforms_name=='RandomRotate':
        transforms_model = [RandomRotate(25)]
    elif transforms_name=='RandomShear':
        transforms_model = [RandomShear(0.15)]
    elif transforms_name=='RandomHSV':
        transforms_model = [RandomHSV(100, 100, 100)]
    else:
        print('wrong')




    transforms = Sequence(transforms_model)

    img, bboxes = transforms(img, bboxes)


    if np.isnan(bboxes).any():
        print('bbboxes',bboxes,file_name)
        return df_all
    elif len(bboxes)==0:###这个判断为空的有效
        print('bbboxes',bboxes,file_name)
        return df_all
    else:
        print('filename2={},class2={}'.format(file_name, folder_name))
        print('bbboxes not null', bboxes)

        'filename,width,height,class,xmin,ymin,xmax,ymax'


        df_temp = pd.DataFrame(data=bboxes, columns=['xmin', 'ymin', 'xmax', 'ymax'])

        file_name='{}_times{}_{}'.format(transforms_name,times,file_name)



        df_temp['filename'] = file_name
        df_temp['folder'] = folder_name

        # print('''dd['class'].values.tolist()''',dd['class'].values.tolist())
        df_temp['class'] = dd['class'].values.tolist()  # 要把转为list才直接赋值给一个新的dataframe列
        print('df_temp', df_temp)
        df_all = df_all.append(df_temp)
        if not os.path.exists('raw_augment/{}'.format(folder_name)):
            os.makedirs('raw_augment/{}'.format(folder_name))
            mp.imsave('raw_augment/{}/{}'.format(folder_name, file_name), img)
        else:
            mp.imsave('raw_augment/{}/{}'.format(folder_name,file_name), img)
        return df_all


def data_aug():

    ###要根据csvs生成jpg路径从而读取
    rootpath_jpg=r'E:\tsl_file\windows_v1.8.1\windows_v1.8.1\label_data_self\raw'
    g = os.walk(rootpath_jpg)
    index=0
    df_all=pd.DataFrame()

    ###可能存在jpg图片存在，但是对于的xml是没有的，也就是对于的csv文件是没有这个图片的，所有要先判断一下
    for path,dir_list,file_list in g:
        # if index==2:
        #     break
        for file_name in file_list:
            try :
                # print(os.path.join(path, file_name) )
                folder_name=path.split('\\')[-1]
                if folder_name=='':
                    print('path00',path)
                    continue

                dd = df[(df['folder'] == folder_name) & (df['filename'] == file_name)]
                ##jpg图片在csv中没有时，dd应该为空
                if dd.empty:
                   print('folder_name/file_name:{}/{}'.format( folder_name,file_name))
                   continue
                print('dd',dd)
                if True in pd.isnull(dd).any():
                    print('dd2',dd,file_name)
                    continue


                # print('dd_shape={},dd={}'.format(dd.shape,dd))
                print(path,file_name)
                bboxes=np.array(dd[['xmin', 'ymin', 'xmax', 'ymax']].values,dtype=np.float64)
                print('bboxess',bboxes)

                #nan进入下一次循环
                if np.isnan(bboxes).any():
                    print('bboxes',bboxes,file_name)
                    continue

                # print('filename={},class={}'.format(file_name, class_name))
                # print('bboxes_shape={},bboxes={}'.format(bboxes.shape, bboxes))
                image_path=os.path.join(path,file_name)
                print('image_path',image_path)
                img=cv2.imread(image_path)[:,:,::-1] #OpenCV uses BGR channels
                # print('img',img)
                # img = Image.open(image_path)
                # img_o=img
                ##经过变换之后框可能变少；这些情况统统不要；变换之后还可能坐标越界这个就try-except跳过了
                if bboxes.shape[0] != dd.shape[0]:
                    continue
                    # # cctv10就不一样
                    # print('wrong', dd.shape[0], bboxes.shape[0])
                    # plt.imshow(img_o)
                    # plt.show()
                    #
                    # # print('bboxes2_shape={},bboxes2={}'.format(bboxes.shape,bboxes))
                    # aug_img=draw_rect(img, bboxes)
                    # plt.imshow(aug_img)
                    # plt.show()

                '''
                RandomHorizontalFlip(0.1) #水平翻转#应该是沿着水平线左右翻转
                RandomScale(0.3, diff = True)#缩放
                RandomTranslate(0.2, diff = True)#平移
                RandomRotate(25)#旋转
                RandomShear(0.2) #随机剪切
                ###Resize(1000) #不使用，设置成总维度为1000
                RandomHSV(100, 100, 100)##组合时一组不加randomhsv,一组加randomhsv;每组都运行5次
                
                
                ####先使用单个的每个生成图片然后在使用组合的生成图片
                ###每个都运行5次##单个的每个运行5次，sequence的运行30次
                单个时RandomHorizontalFlip(0.8)单个时设置0.8，组合时设置成0.1               
                '''
                #单个的执行5次

                for transforms_name in ['RandomHorizontalFlip','RandomScale','RandomTranslate',
                                            'RandomRotate','RandomShear','RandomHSV']:
                    for times in range(5):
                         df_all=each_jpg_transform(transforms_name, df_all, img, bboxes, file_name, folder_name, dd, times)


                for transforms_name in ['sequence1_model','sequence2_model']:##senquence执行30
                     for times in range(30):
                          df_all=each_jpg_transform(transforms_name, df_all, img, bboxes, file_name, folder_name, dd, times)


            except Exception as e:
                print('my err',e)




        # index=index+1
    df_all.to_csv('train_labels_new.csv',index=False)
    print(df_all)


if __name__=='__main__':
    data_aug()

