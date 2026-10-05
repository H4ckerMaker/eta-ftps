# Encrypted Traffic Analysis for FTPS connection
## Related Work in the topic
Previous research regarding encrypted network traffic analysis has shown that using packet sizes and inter-arrival times as classification parameters only required 10-30 packets and resulted in sufficient classification accuracy (Roy, Shapira & Shavitt, 2022). Similarly, another research investigates traffic classification regarding flow features including packet sizes, inter-arrival times and packet directions (Lu, G. et al., 2012). 
## What applies to our project
Since the collected data from the FTPS traffic for this project is encrypted, the classification cannot be done using the payload but must instead rely on the characteristic of the traffic. Therefore, the aim of this project is to extract packet sizes and inter-arrival times as the features to be analysed. Although the previous research do not directly focus on FTPS, they still provide guidance regarding what features to extract for the classification. 
Packet size is relevant to this project since the same file size may still have packets of different sizes. Therefore, the distribution of packet sizes might provide necessary information in contrast to the total size of the file. Inter-arrival time, the time interval between the packets, may also record differences between the files and make it possible to differentiate the files.
The packet sizes and inter-arrival times will thus be the features extracted from the FTPS interaction to be used for the classification. 

