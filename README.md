# Encrypted Traffic Analysis for FTPS connection
## Short Description of the Project/ Documentation of the Project
The goal of our project is to classify encrypted network traffic of different protocols such as file transfers, browsing, chatting and streaming (videos and audio recordings). This was done by capturing multiple files of each protocol and thereafter training and testing two different models (one simpler and one more complex).  

The parameters used for classification were selected by reading previous research and reports at KTH Library regarding the subject and choosing the parameters used in the papers most applicable to our project. Information regarding the previous research used for the project can be found further down in the README. 

With the complex model, we achieved an accuracy of 90% on the data provided by us. 
## How to Run the Code/ Documentation of Testing the Project

## Previous Research
Previous research regarding encrypted network traffic analysis has shown that using packet sizes and inter-arrival times as classification parameters only required 10-30 packets and resulted in sufficient classification accuracy (Roy, Shapira & Shavitt, 2022). Similarly, another research investigates traffic classification regarding flow features including packet sizes, inter-arrival times and packet directions (Lu, G. et al., 2012). 
## How It Applies to Our Project
Since the collected data from the FTPS traffic for this project is encrypted, the classification cannot be done using the payload but must instead rely on the characteristic of the traffic. Therefore, the aim of this project is to extract packet sizes and inter-arrival times as the features to be analysed. Although the previous research do not directly focus on FTPS, they still provide guidance regarding what features to extract for the classification. 
Packet size is relevant to this project since the same file size may still have packets of different sizes. Therefore, the distribution of packet sizes might provide necessary information in contrast to the total size of the file. Inter-arrival time, the time interval between the packets, may also record differences between the files and make it possible to differentiate the files.
The packet sizes and inter-arrival times will thus be the features extracted from the FTPS interaction to be used for the classification. 
## References
Roy, S., Shapira, T., & Shavitt, Y. (2022). Fast and lean encrypted Internet traffic classification. Computer Communications, 186, 166–173. https://doi.org/10.1016/j.comcom.2022.02.003

Lu, G. et al. (2012) ‘Comparison and Analysis of Flow Features at the Packet Level for Traffic Classification’, in 2012 International Conference on Connected Vehicles and Expo (ICCVE). [Online]. December 2012 IEEE. pp. 262–267.

