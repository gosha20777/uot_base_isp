from torch import nn

def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()    
    
def unfreeze(model):
    for p in model.parameters():
        p.requires_grad_(True)
    model.train(True)

def freeze_T_layers(model, resolution):
    # resolution - power, i.e 5 for 32x32
    cnt_to_freeze = resolution - 2
    for child in model.children():
        total_cnt = len(child)
        tmp_cnt = 0
        for ch in child:
            if total_cnt - tmp_cnt <= cnt_to_freeze:
                #freeeeeeeeeeze
                for param in ch.parameters():
                    param.requires_grad_(False)
            tmp_cnt += 1

    model.train(True)

def freeze_D_layers(model, resolution):
    cnt_to_freeze = resolution - 2
    for i, child in enumerate(model.children()):
        if i == 0:
            total_cnt = len(child)
            tmp_cnt = 0
            for ch in child:
                if total_cnt - tmp_cnt <= cnt_to_freeze:
                    #freeeeeeeeeeze
                    for param in ch.parameters():
                        param.requires_grad_(False)
                tmp_cnt += 1

        if i == 2:
            total_to_freeze = 2 + (cnt_to_freeze - 1) * 3
            total_cnt = len(child)
            tmp_cnt = 0
            for ch in child:
                if total_cnt - tmp_cnt <= total_to_freeze:
                    #freeeeeeeeeeze
                    for param in ch.parameters():
                        param.requires_grad_(False)
                tmp_cnt += 1

        model.train(True)

def weights_init_D(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        
def weights_init_mlp(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')