clear ; close all; clc
centroids = [
   0.655670   0.475347   0.217509;
   0.081205   0.087875   0.077120;
   0.431500   0.466058   0.568975;
   0.442656   0.323410   0.221242;
   0.927677   0.879655   0.763905;
   0.132308   0.142398   0.136862;
   0.587590   0.499047   0.453985;
   0.165334   0.175665   0.179514;
   0.861625   0.734906   0.507947;
   0.302402   0.299245   0.318302;
   0.897872   0.624641   0.313828;
   0.060233   0.065912   0.056068;
   0.238124   0.217447   0.206746;
   0.191789   0.166982   0.126138;
   0.725973   0.619245   0.447929;
   0.106498   0.114709   0.104636
];
figure;
rgbplot (centroids, "composite");
figure;
cmap = hsv (256);
rgbplot (cmap, "composite");
%subplot(1, 2, 2);

pause;