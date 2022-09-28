1. AffectNetData===is  around 10GB and can be downloaded from [here](http://mmlab.ie.cuhk.edu.hk/projects/AffectNet/AffectNet-Manual.pdf).
  It is a large-scale facial expression database with 7 emotion labels and 32,589 images.
    The images are collected from YouTube videos and labeled by crowd-sourcing.
    The images are cropped and aligned by the authors.
    The images are in the size of 64x64.
    The images are in the format of .jpg.
    The images are in the range of [0, 255].
    The images are in the shape of (64, 64, 3).
    The labels are in the range of [0, 6].
    The labels are in the shape of (1,).
    The labels are in the format of .csv.

2. Django-with-celery-tasks:In this mainly the working of proctoring  and mobile phone detection and 
    celery mainly used for  the background task if  user taking  more time to process the whole file by model 
    that time we use mainly celery for background task.

3.  Object-Detection using yolov7 : In this mainly we do the mobile-phone detetction using yolov7 fine tune the model on  
    custom dataset and  we use the yolov7 for mobile phone detection and we use the yolov7 for proctoring.
    if refer than use simple roboflow working  directory for  finetune model  for training and testing.

4. RepVGG : In this mainly we do frame level emotion in  8 class ,Search for official implementation they used 
    yolov7 with RepVGG ,
5. Speech-wav2vec2.0 : This is currently under development and will be updated soon as possible.



6. Wav2vec+SVM : Embeddings are extracted from wav2vec2.0 and then SVM is used for classification.
    accuracy not good but it is good fluecny level but hope it will be good 