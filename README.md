<<<<<<< HEAD
Dataset = https://www.kaggle.com/datasets/simuletic/cctv-weapon-dataset
Annotation website = https://labelstud.io/

Notebook I used for reference:
https://colab.research.google.com/github/EdjeElectronics/Train-and-Deploy-YOLO-Models/blob/main/Train_YOLO_Models.ipynb#scrollTo=4letvP7X12ji

THe folders runs\detect will not appear unless you have ran the training code for the first time.

## June 12, 2026 Training-

First training with 206 images yielded over labelling. May need more images. So far on an Processor	12th Gen Intel(R) Core(TM) i5-12450H, 2000 Mhz, 8 Core(s), 12 Logical Processor(s) the traning took 2min.

## **July 16 Training** -

80/20 split - Seed 42


```python
model.train(
    data         = 'datasets/data.yaml',  # path to our config file
    epochs       = 75,                   # maximum number of training epochs
    patience     = 75,                    # stop early if no improvement for 75 epochs
    imgsz        = 640,                   # resize all images to 640x640 for training
    batch        = 8,                     # images per training batch (RTX 4050: 8-12 recommended)
    device       = 0,                     # ✅ use GPU device 0 (your RTX 4050)
    optimizer    = 'SGD',                 # SGD optimizer — stable for small datasets
    lr0          = 0.001,                 # initial learning rate
    lrf          = 0.01,                  # final lr = lr0 * lrf (cosine decay)
    warmup_epochs= 5,                     # gradually increase lr for first 5 epochs
    weight_decay = 0.0005,                # regularization to prevent overfitting
    augment      = True,                  # apply random augmentation during training
    mosaic       = 1.0,                   # mosaic augmentation (combines 4 images)
    fliplr       = 0.5,                   # 50% chance of horizontal flip per image
    hsv_h        = 0.015,                 # hue shift range
    hsv_s        = 0.7,                   # saturation jitter
    hsv_v        = 0.4,                   # brightness jitter
    pretrained   = True,                  # start from ImageNet pretrained weights
    name         = 'gun_detector'         # folder name for saving results
)
```


**Training Time: 00:41:59.1**

Accuracy:  0.8358
Precision: 1.0000
Recall:    0.8358
F1 Score:  0.9106

Best mAP50: 0.9365 at epoch 9

    epoch  train/box_loss  train/cls_loss  metrics/mAP50(B)
0       1         1.93789         3.76268           0.44511
1       2         1.73587         1.92397           0.78714
2       3         1.59533         1.39849           0.88533
3       4         1.50000         1.18805           0.90899
4       5         1.36581         0.97826           0.91204
5       6         1.32008         0.91968           0.90242
6       7         1.21112         0.80403           0.86744
7       8         1.14446         0.72065           0.91473
8       9         1.10931         0.68693           0.93648
9      10         1.03556         0.63378           0.91767
10     11         0.98900         0.62022           0.89607
11     12         0.96875         0.59418           0.88902
12     13         0.95365         0.58648           0.87596
13     14         0.90550         0.53829           0.89131
14     15         0.87994         0.55600           0.86474
15     16         0.84914         0.52223           0.89871
16     17         0.82143         0.51063           0.89026
17     18         0.80224         0.49193           0.90247
18     19         0.79803         0.48416           0.88148
19     20         0.77410         0.47593           0.89942
20     21         0.74654         0.46798           0.88944
21     22         0.74440         0.46657           0.88525
22     23         0.70405         0.44156           0.88467
23     24         0.69708         0.43247           0.91125
24     25         0.69142         0.43734           0.89140
25     26         0.67553         0.41378           0.88573
26     27         0.65045         0.40445           0.88971
27     28         0.66278         0.41185           0.87332
28     29         0.63407         0.39710           0.86891
29     30         0.64168         0.40533           0.87942
30     31         0.62967         0.39007           0.90560
31     32         0.62534         0.39794           0.91028
32     33         0.62274         0.38839           0.87068
33     34         0.59178         0.37560           0.85445
34     35         0.60139         0.38119           0.85708
35     36         0.59632         0.38047           0.85718
36     37         0.58573         0.37167           0.88993
37     38         0.57664         0.35926           0.85372
38     39         0.56981         0.35191           0.86420
39     40         0.56259         0.35029           0.89201
40     41         0.57394         0.35995           0.86080
41     42         0.56140         0.34908           0.85410
42     43         0.55375         0.35845           0.87413
43     44         0.54156         0.34197           0.89059
44     45         0.53617         0.35020           0.86639
45     46         0.53282         0.34182           0.89498
46     47         0.53600         0.33626           0.89835
47     48         0.51834         0.32320           0.89852
48     49         0.52066         0.33500           0.89113
49     50         0.51637         0.32797           0.87220
50     51         0.50984         0.33069           0.88091
51     52         0.50515         0.32461           0.85155
52     53         0.49347         0.31311           0.86624
53     54         0.50865         0.32172           0.85621
54     55         0.50487         0.32104           0.84925
55     56         0.48833         0.32159           0.85168
56     57         0.47792         0.31807           0.85593
57     58         0.49466         0.31648           0.87116
58     59         0.47026         0.30283           0.86415
59     60         0.47632         0.30508           0.84645
60     61         0.47434         0.30739           0.86487
61     62         0.49340         0.32508           0.86025
62     63         0.47483         0.30416           0.86891
63     64         0.46246         0.29959           0.88050
64     65         0.45323         0.29591           0.84025
65     66         0.39764         0.23417           0.84728
66     67         0.39350         0.22857           0.83563
67     68         0.37938         0.22357           0.83518
68     69         0.36330         0.21207           0.83651
69     70         0.36204         0.21362           0.83955
70     71         0.35759         0.21058           0.83970
71     72         0.36337         0.21554           0.83967
72     73         0.34741         0.21089           0.83773
73     74         0.35742         0.23813           0.84242
74     75         0.34879         0.20724           0.83942

In gun identification—whether for security cameras, baggage scanning, or forensic audio— =a **Precision of 1.0000** means the system has **zero false alarms**= . Every time the model alerts security that a gun is present, there is a gun there.

However, because your  **Recall is 0.8358** , the model is currently  **missing 16.4% of all guns** . In a security or safety environment, letting roughly 1 in 6 guns pass through undetected is a massive safety hazard.

For gun identification, you almost always want to prioritize **high Recall** over perfect Precision. It is much safer to investigate a few false alarms than to miss an actual weapon.

**Errors Encountered**:

Multiple running instances may lead to file overlap and cause an error at cell 28: The viewing of results. - Resolved - Restarting Kerne

## **July 18 Training -**

80/20 split- Random Seed

```python
model.train(
    data         = 'datasets/data.yaml',  # path to our config file
    epochs       = 75,                   # maximum number of training epochs
    patience     = 75,                    # stop early if no improvement for 75 epochs
    imgsz        = 640,                   # resize all images to 640x640 for training
    batch        = 8,                     # images per training batch (RTX 4050: 8-12 recommended)
    device       = 0,                     # ✅ use GPU device 0 (your RTX 4050)
    optimizer    = 'SGD',                 # SGD optimizer — stable for small datasets
    lr0          = 0.001,                 # initial learning rate
    lrf          = 0.01,                  # final lr = lr0 * lrf (cosine decay)
    warmup_epochs= 5,                     # gradually increase lr for first 5 epochs
    weight_decay = 0.0005,                # regularization to prevent overfitting
    augment      = True,                  # apply random augmentation during training
    mosaic       = 1.0,                   # mosaic augmentation (combines 4 images)
    fliplr       = 0.5,                   # 50% chance of horizontal flip per image
    hsv_h        = 0.015,                 # hue shift range
    hsv_s        = 0.7,                   # saturation jitter
    hsv_v        = 0.4,                   # brightness jitter
    pretrained   = True,                  # start from ImageNet pretrained weights
    name         = 'gun_detector'         # folder name for saving results
)
```


**Training Time: 00:42:03:03**

Accuracy:  0.8358
Precision: 1.0000
Recall:    0.8358
F1 Score:  0.9106

Best mAP50: 0.9412 at epoch 13

    epoch  train/box_loss  train/cls_loss  metrics/mAP50(B)
0       1         1.95040         3.71470           0.46805
1       2         1.76774         1.92140           0.76331
2       3         1.66207         1.42578           0.83973
3       4         1.54543         1.13854           0.87043
4       5         1.42564         1.00075           0.91721
5       6         1.33643         0.88074           0.90577
6       7         1.24358         0.78421           0.93337
7       8         1.17182         0.72276           0.91187
8       9         1.11563         0.69633           0.88723
9      10         1.05989         0.65689           0.89587
10     11         1.01533         0.62650           0.92411
11     12         0.96907         0.59457           0.92734
12     13         0.94524         0.57337           0.94120
13     14         0.91816         0.54462           0.93604
14     15         0.90024         0.55836           0.88918
15     16         0.84521         0.52045           0.90469
16     17         0.85635         0.51588           0.90680
17     18         0.80515         0.49589           0.88544
18     19         0.79065         0.48185           0.88296
19     20         0.77887         0.48414           0.90431
20     21         0.76625         0.46270           0.89463
21     22         0.74925         0.46715           0.92919
22     23         0.71719         0.43487           0.92667
23     24         0.74290         0.45350           0.89070
24     25         0.69990         0.43491           0.88397
25     26         0.70018         0.44070           0.87160
26     27         0.69161         0.41933           0.90684
27     28         0.66729         0.41045           0.91899
28     29         0.65949         0.40348           0.88962
29     30         0.65863         0.41605           0.91229
30     31         0.64435         0.40814           0.89289
31     32         0.62972         0.39505           0.88364
32     33         0.63132         0.39590           0.90996
33     34         0.62605         0.38438           0.88491
34     35         0.61326         0.37767           0.87593
35     36         0.59061         0.37416           0.86136
36     37         0.61081         0.38724           0.84372
37     38         0.59610         0.37703           0.87296
38     39         0.57244         0.35597           0.88308
39     40         0.57421         0.36305           0.88334
40     41         0.57795         0.35423           0.90320
41     42         0.57219         0.36255           0.92121
42     43         0.55970         0.35352           0.90886
43     44         0.56523         0.35807           0.89817
44     45         0.57117         0.35175           0.88560
45     46         0.53401         0.34008           0.88953
46     47         0.55427         0.34603           0.88293
47     48         0.53825         0.33605           0.90269
48     49         0.53468         0.34156           0.88053
49     50         0.51078         0.31767           0.87578
50     51         0.52303         0.33767           0.87969
51     52         0.52252         0.32896           0.85890
52     53         0.50066         0.31270           0.88441
53     54         0.49173         0.31066           0.86159
54     55         0.49595         0.31126           0.89023
55     56         0.50629         0.32738           0.87489
56     57         0.50343         0.32980           0.85126
57     58         0.48870         0.30887           0.85508
58     59         0.49740         0.31311           0.86827
59     60         0.48267         0.30702           0.86175
60     61         0.48584         0.30860           0.83400
61     62         0.48058         0.30568           0.83953
62     63         0.46416         0.29876           0.85181
63     64         0.47365         0.30844           0.86176
64     65         0.47060         0.29715           0.85498
65     66         0.41667         0.23354           0.87964
66     67         0.39355         0.22355           0.84961
67     68         0.38589         0.22920           0.84709
68     69         0.37346         0.21087           0.85198
69     70         0.37672         0.21212           0.85645
70     71         0.36951         0.20870           0.86869
71     72         0.36278         0.20773           0.87616
72     73         0.35900         0.20454           0.87516
73     74         0.36638         0.20812           0.87244
74     75         0.36452         0.21670           0.86195


That orange dot at epoch 13 is your best model — and everything after it is a warning sign, not progress.

**The core issue: you trained way past the point of diminishing returns**

mAP50 hit its peak (0.9412) at epoch 13. After that it never beats that number again — it oscillates in the 0.83–0.93 range for the remaining 62 epochs while slowly drifting *down* overall, ending at 0.862. Meanwhile your training losses (box_loss, cls_loss) keep dropping smoothly the entire time. That divergence — train loss still falling, validation mAP flat-to-declining — is the textbook signature of **overfitting**: the model is increasingly memorizing your training images rather than generalizing.

**Why this happened: your early-stopping never had a chance to fire**

You set `patience = 75` with `epochs = 75`. Patience needs to be *less* than total epochs to actually stop training early — with patience equal to epochs, YOLO will never trigger it, so training just runs the full schedule regardless of whether performance improved. Effectively you disabled early stopping.

**One more clue in the loss curve**

There's a sharp discontinuity at epoch 66 — both box_loss and cls_loss suddenly drop (0.480→0.417 and 0.299→0.234). This lines up with Ultralytics' default `close_mosaic=10` behavior, which turns off mosaic augmentation for the final 10 epochs. Training on "easier," non-mosaiced images makes the loss look better, but it doesn't translate to a validation mAP improvement — another sign the model wasn't meaningfully learning anything new in that final stretch.

**About the Accuracy/Precision/Recall/F1 block**

Those numbers were almost certainly computed from `last.pt` (the final epoch's weights) rather than `best.pt` (epoch 13, which YOLO saves automatically). Since mAP50 at the final epoch (~0.86) is meaningfully worse than the peak (0.94), your reported recall of 0.836 is likely understating what your actual best checkpoint can do.

**What I'd do next**

1. **Evaluate `best.pt`, not `last.pt`** — recompute precision/recall/accuracy from the epoch-13 weights. This is likely a quick win.
2. **Fix patience** — set it to something like 15–20 so training stops automatically once mAP50 stops improving. You'll save GPU time too (this run took 42 minutes for a plateau that was reached in the first third).
3. **Check dataset size** — with a peak this early and this much epoch-to-epoch volatility, your training/validation sets may be fairly small. More diverse data (varied angles, lighting, gun types, backgrounds, partial occlusions) would likely help more than more epochs.
4. **Investigate the recall gap on `best.pt` specifically** — pull the confusion matrix or look at which validation images are missed to see if it's a particular gun type, size, or pose driving the false negatives.

If you can share the full `results.csv` (it likely has `metrics/precision`, `metrics/recall`, and `mAP50-95` columns too) or run validation directly on `best.pt`, I can help you pin down the real performance more precisely.

---
=======
September 26, 2026 Added new pictures without guns for the validation set. Used latest trained run for testing. 
Accuracy:  0.8955
Precision: 0.9492
Recall:    0.8358
F1 Score:  0.8889

September 15, 2026: 
Forked from the collaborated original.
Will add basic frontend UI for Upload and basic image detection through Onnx 

Dataset = https://www.kaggle.com/datasets/simuletic/cctv-weapon-dataset
Annotation website = https://labelstud.io/

Notebook I used for reference: 
https://colab.research.google.com/github/EdjeElectronics/Train-and-Deploy-YOLO-Models/blob/main/Train_YOLO_Models.ipynb#scrollTo=4letvP7X12ji

THe folders runs\detect will not appear unless you have ran the training code for the first time. 

June 12, 2026 Training
First training with 206 images yielded over labelling. May need more images. So far on an Processor	12th Gen Intel(R) Core(TM) i5-12450H, 2000 Mhz, 8 Core(s), 12 Logical Processor(s) the traning took 2min.



>>>>>>> 60081a45a8adc8f92ee66516aea7411f685a4aa2
