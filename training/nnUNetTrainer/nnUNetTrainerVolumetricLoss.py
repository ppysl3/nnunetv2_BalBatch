from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.compound_losses import  DC_and_BCE_loss
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.dice import SoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
import numpy as np
import torch
from torch import autocast, nn
from typing import Callable


class VolumetricLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True, clip_tp: float = None):
        """
        """
        super(VolumetricLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.clip_tp = clip_tp
        self.ddp = ddp

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape
        print("Shape x "+str(shp_x))
        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
        #targetcopy=y.cpu().detach().numpy()
        #targetsum=np.sum(targetcopy)
        #print("yshape " +str(y.shape))
        #targetsum=torch.sum(y)   
        #print("Targetsum "+str(targetsum)) 
        #print(targetcopy)
        tp, fp, fn, tn = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)
        print("Total Pixels  " +str(tp+fp+fn+tn))
        if self.ddp and self.batch_dice:
            tp = AllGatherGrad.apply(tp).sum(0)
            fp = AllGatherGrad.apply(fp).sum(0)
            fn = AllGatherGrad.apply(fn).sum(0)

        if self.clip_tp is not None:
            tp = torch.clip(tp, min=self.clip_tp , max=None)

        numyes=(tp+fp)
        print("Allyes "+str(numyes))
        hypotheticaltarget=(tp+fn)
        print("ShouldBeEqualToTarget "+str(hypotheticaltarget))
        #numyes=np.array(numyes.detach().cpu().numpy())
        #numyes=np.array(numyes[1])
        #numyes=numyes[1]
        #print("Num yes " +str(numyes))
        #print("TargetSum" +str(targetsum))
        #print("Num Predicted Pixels" +str(numyes))
        
        Diff=hypotheticaltarget-numyes
        PotentialLoss=50*((Diff/(tp+fp+fn+tn)))

        #Diff=targetsum-numyes
        #checksize=np.prod(x.shape)
        #PotentialLoss=100*((Diff)/(checksize))**2 #This is squared
        print("Potential Loss" +str(PotentialLoss))
        print("Hopefully One Column " +str(PotentialLoss[:,0]))
        PotentialLoss=torch.square(PotentialLoss[:,0]).mean()
        #import sys
        #sys.exit()
        return PotentialLoss

class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        
        #TEMPORARY FOR TEST
        #dice_class=VolumetricLoss

        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.vl=VolumetricLoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0].long()) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0
        vl_loss = self.vl(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        print("DICE LOSS "+str(dc_loss))
        print("CE LOSS "+str(ce_loss))
        print("VOL LOSS"+ str(vl_loss))
        print("")
        #print("Loss Outside" +str(vl_loss))
        outputnp=net_output.cpu().detach().numpy()
        targetnp=target.cpu().detach().numpy()
        outputnp=np.array(outputnp)
        targetnp=np.array(targetnp)
        #Issue with doing this in the loss is the deep supervision, it'll do it across all layers one by one.
        #print("")
        #print("WithinLossShape")
        #print(outputnp.shape)
        #if float(np.sum(outputnp[:,0,:,:])).is_integer():
        #    print("Am Int")
        #print(np.sum(outputnp[:,0,:,:]))
        #print(targetnp.shape)
        #print(np.sum(targetnp[:,0,:,:]))
        dc=(dc_loss.cpu().detach().numpy())
        #print("DICE" +str(dc))
        print("CE WEIGHT "+str(self.weight_ce))
        print("DICE WEIGHT" +str(self.weight_dice))
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss + 1*vl_loss
        return result





class nnUNetTrainerVolumetricLoss(nnUNetTrainer):
    def train_step(self, batch: dict) -> dict:
        print("")
        print("TrainStep")
        data = batch['data']
        target = batch['target']
        fullsize=(target[4]) #0 For fullsize
        fullsize=fullsize.cpu().detach().numpy()
        #print(fullsize.shape)

        #print("SUM - "+str(np.sum(fullsize[:,0,:,:])))
        #print("Target as loaded"+str([i.size() for i in target]))
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad()
        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            #print("CAST")
            #print("DATASHAPE" +str([i.size() for i in data]))
            #print("TargetShape" +str([i.size() for i in target]))
            print("SendThroughNetwork")
            output = self.network(data)
            #This wont work, but the error message gives us the sizes so. Del data was there before I started editing
            #newout=torch.stack(output)
            #newout=torch.tensor(newout, device='cpu')
            #newtarg=torch.stack(target)
            #newtarg=torch.tensor(target, device='cpu')
            # del data
            
            fullsizeout=(output[4]) #0 for Full Size
            fullsizeout=fullsizeout.cpu().detach().numpy()
            #print("Fullsized out"+str(fullsizeout.shape))
            #print("SUMZERO="+str(np.sum(fullsizeout[:,0])))
            #print("SUMONE="+str(np.sum(fullsizeout[:,1])))
            #print("Target out"+str(fullsize.shape))
            #print("SumTargetZero="+str(np.sum(fullsize[:,0])))
            #print("")
            print("SendForLoss")
            l = self.loss(output, target)
            copyl=l.cpu().detach().numpy()
            #print("GotLoss " + str(copyl))
        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}
    def _build_loss(self):
        if self.label_manager.has_regions:
            raise Exception("Volumetric Loss Function currently not supported in region based training.")
            import sys
            sys.exit()
            loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
                                   use_ignore_label=self.label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientSoftDiceLoss)
        else:
            print("LoadingNewLoss")
            loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                                  ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)

        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)
        return loss
