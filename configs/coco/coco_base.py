model = dict(
    name="base",
    num_classes=80
)

useCopyDecoupling = False
dataset = dict(
    name = "coco",
    root = "/home/share1/coco/",
    useCopyDecoupling = useCopyDecoupling,
    sampler='ClassAware',
    clsDataListFile = '/home/pengpeng/LongTailMLC/appendix/coco/longtail2017/class_freq.pkl',
    imageSize=224,
)

loss = dict(
    name="asl",
    param = dict(
    gamma_neg = 100,
    gamma_pos = 0,
    clip = 0.05,
    disable_torch_grad_focal_loss = True,
    useCopyDecoupling = useCopyDecoupling
    )
)

epochs = 80

output_path = "./{}_{}_ClassAware".format(dataset['name'],loss['name'],)


