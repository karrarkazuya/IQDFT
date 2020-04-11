# IQDFT
Face tool made with opencv in python

### Features

- Supports Face bluring or adding a mask (image) on faces;

| Blur face hiding 1 |  Blur face hiding 2 |
| --- | --- |
|![](https://github.com/karrarkazuya/IQDFT/blob/master/redme_media/blur_1.gif) | ![](https://github.com/karrarkazuya/IQDFT/blob/master/redme_media/blur_2.gif) |

- Supports Faces swapping of two videos;

| Source video |  Target video |  Result video |
| --- | --- | --- |
|![](https://github.com/karrarkazuya/IQDFT/blob/master/redme_media/swap_1.gif) | ![](https://github.com/karrarkazuya/IQDFT/blob/master/redme_media/swap_2.gif) | ![](https://github.com/karrarkazuya/IQDFT/blob/master/redme_media/swap_3.gif) |



                
----


This tool is using a higly modifid version of [wuhuikai FaceSwap ](https://github.com/wuhuikai/FaceSwap)way of implementing face swap so check that out too


#### Requirements

use pip in python to install the following

opencv
`$ pip install opencv-python`

cmake
`$ pip install cmake`

dlib
`$ pip install dlib`

numpy
`$ pip install numpy`

scipy
`$ pip install scipy`

#### Main functions

you can blur up faces in a video like this

    IQDFT.hide_faces("video_1.mp4", "new_video.mp4")
or you can instead use a picture on the faces instead of blur

    IQDFT.hide_faces("video_1.mp4", "new_video.mp4", "smile.png")
you can extract faces (if you want) like this, this will cut and save each face in a video

    IQDFT.extract_faces("video_1.mp4", "faces")
you can also swap faces between videos like this

    to_add_video = "video_1.mp4"
    original_video = "video_2.mp4"
    output_video = "new_video.mp4"
    IQDFT.swap_faces(to_add_video, original_video, output_video)
    
And thats basically it!


#### Worth to mention　

This tool comes with no warranty or guarantee of any kind, the face detection might not always detect faces to blur so you will always have to check the videos after that to check if any frames went without blurring or hiding. also the face swap on videos is for educational cases so you could learn about how this works and all and you are not allowed to use it in any way to fake videos to harm people.
also the videos I used are already on YouTube and here are some links
[Video1](https://www.youtube.com/watch?v=Vw1JqbvgZCc)
[Video2 ](https://www.youtube.com/watch?v=C-Zj0tfZY6o)
[Video3 ](https://www.youtube.com/watch?v=zkIfCo2JxJY)
enjoy!
