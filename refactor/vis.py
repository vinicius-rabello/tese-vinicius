import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from DSCMS import DSCMS
from loader import SuperResNpyDataset2
import torch
import torch.nn.functional as F
import sys

from DSCMS import DSCMS

from metrics.image_metrics import ssimLoss, psnrLoss, lpipsLoss, fsimLoss, epiLoss
from metrics.physical_metrics import LossTKE

year = int(sys.argv[1])
day = int(sys.argv[2])

lowNP = np.load(f"data/100/window_{year:04d}.npy")
lat = lowNP[day,:,:,0]
lon = lowNP[day,:,:,1]
lowNP_u = lowNP[day,:,:,2]
lowNP_v = lowNP[day,:,:,3]
lowNP_o = lowNP[day,:,:,4]/1.0e-5

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = DSCMS(2,2,3)
model = torch.nn.DataParallel(model)
model = model.to(device)

# random window
mean = np.array([-0.00561308,0.07556629])
std = np.array([0.32576539,0.38299691])

model.load_state_dict(torch.load('./25.pth', map_location=torch.device(device)))    #  2D_MOdel1 (even numbers)

data_folder = "./data"
lr_files = [f"25/window_{year:04d}.npy"]
hr_files = [f"100/window_{year:04d}.npy"]

dataset = SuperResNpyDataset2(data_folder, lr_files, hr_files,0,mean,std)
test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

vmin = 0.0 #e-5
vmax = 1.0 #e-5

trim = 4

lowMin,lowMax   =  -11.453426448716854,23.47178113961668
highMin,highMax =  -27.36551687121391,45.74970516841858

bilinear_MSE_u,bilinear_MSE_v,bilinear_MSE_o = [],[],[]
bic_MSE_u,bic_MSE_v,bic_MSE_o = [],[],[]
bic_SSIM_u,bic_SSIM_v = [],[]
bic_PSNR_u,bic_PSNR_v = [],[]
model_PSNR_u,model_PSNR_v = [],[]
LR_lpips_u,LR_lpips_v = [],[]
model_lpips_u,model_lpips_v = [],[]
bilinear_lpips_u,bilinear_lpips_v =  [] , []
bic_lpips_u,bic_lpips_v =  [] , []
LR_FSIM_u,LR_FSIM_v = [],[]
model_FSIM_u,model_FSIM_v = [],[]
bilinear_FSIM_u,bilinear_FSIM_v =  [] , []
bic_FSIM_u,bic_FSIM_v =  [] , []
LR_EPI_u,LR_EPI_v = [],[]
model_EPI_u,model_EPI_v = [],[]
bilinear_EPI_u,bilinear_EPI_v =  [] , []
bic_EPI_u,bic_EPI_v =  [] , []
LR_FID_u,LR_FID_v = [],[]
LR_MSE_u,LR_MSE_v,LR_MSE_o = [],[],[]
LR_SSIM_u,LR_SSIM_v = [],[]
LR_PSNR_u,LR_PSNR_v = [],[]
bilinear_PSNR_u,bilinear_PSNR_v =  [] , []
model_MSE_u,model_MSE_v,model_MSE_o = [],[],[]
reverse_MSE_u,reverse_MSE_v,reverse_MSE_o =[],[],[]
model_SSIM_u,model_SSIM_v,bilinear_SSIM_u,bilinear_SSIM_v = [],[],[],[]
LR_TKE_u,LR_TKE_v = [],[]
model_TKE_u,model_TKE_v = [],[]
bilinear_TKE_u,bilinear_TKE_v =  [] , []
bic_TKE_u,bic_TKE_v =  [] , []

with torch.no_grad():
    for i, (lr, hr) in enumerate(test_loader):
        print(i)
        lr = lr.to(device)  # Move LR to device
        hr = hr.to(device)  # Move HR to device
        loww = lr.cpu().squeeze().numpy()  # lr as a NumPy array (used for debugging)

        # move to CPU and convert to numpy
        lr_np = lr.cpu().numpy()
        lr = torch.tensor(lr_np).to(device)
        lr_upsampled = F.interpolate(lr, scale_factor=4, mode='bicubic', align_corners=False)

        # Get the model output
        output = model(lr_upsampled)
        y = output  # Model's output (not used directly for SSIM)
        print(lr.min(),lr.max(),hr.min(),hr.max(), y.min(),y.max())

        # Convert to numpy for debugging and visualization (don't need to convert back to tensor)
        low = lr_upsampled.cpu().squeeze().numpy()  # lr as a NumPy array (used for debugging)
        high = hr.cpu().squeeze().numpy()  # hr as a NumPy array (used for debugging)
        output = output.cpu().squeeze().numpy()  # hr as a NumPy array (used for debugging)

        # LR Losses   ######################################################################################
        LR_MSE_u.append(np.mean(np.square(high[0,:,:] - low[0,:,:])))
        LR_MSE_v.append(np.mean(np.square(high[1,:,:] - low[1,:,:])))
 
        LR_SSIM_u.append(ssimLoss(hr,lr_upsampled,0,i))
        LR_SSIM_v.append(ssimLoss(hr,lr_upsampled,1,i))

        LR_PSNR_u.append(psnrLoss(hr,lr_upsampled,0,i))
        LR_PSNR_v.append(psnrLoss(hr,lr_upsampled,1,i))

        LR_lpips_u.append(lpipsLoss(hr,lr_upsampled,0,i))
        LR_lpips_v.append(lpipsLoss(hr,lr_upsampled,1,i))

        LR_FSIM_u.append(fsimLoss(hr,lr_upsampled,0,i))
        LR_FSIM_v.append(fsimLoss(hr,lr_upsampled,0,i))

        LR_EPI_u.append(epiLoss(hr,lr_upsampled,0,i))
        LR_EPI_v.append(epiLoss(hr,lr_upsampled,1,i))

        LR_TKE_u.append(LossTKE(hr,lr_upsampled,0,i))
        LR_TKE_v.append(LossTKE(hr,lr_upsampled,1,i))

        # Model Losses ######################################################################################
        model_MSE_u.append(np.mean(np.square(high[0,:,:] - output[0,:,:])))
        model_MSE_v.append(np.mean(np.square(high[1,:,:] - output[1,:,:])))

        model_SSIM_u.append(ssimLoss(y,hr,0,i))
        model_SSIM_v.append(ssimLoss(y,hr,1,i))

        model_PSNR_u.append(psnrLoss(y,hr,0,i))
        model_PSNR_v.append(psnrLoss(y,hr,1,i))

        model_lpips_u.append(lpipsLoss(y,hr,0,i))
        model_lpips_v.append(lpipsLoss(y,hr,1,i))

        model_FSIM_u.append(fsimLoss(y,hr,0,i))
        model_FSIM_v.append(fsimLoss(y,hr,1,i))

        model_EPI_u.append(epiLoss(y,hr,0,i))
        model_EPI_v.append(epiLoss(y,hr,1,i))

        model_TKE_u.append(LossTKE(y,hr,0,i))
        model_TKE_v.append(LossTKE(y,hr,1,i))

        ### bilinear ##############################################################################
        # x = torch.tensor(low.reshape(2,low.shape[1],low.shape[2]), dtype=torch.float32).to(device)
        x = torch.tensor(loww.reshape(2,loww.shape[1],loww.shape[2]), dtype=torch.float32).to(device)
        x8 = F.interpolate(x.unsqueeze(0), scale_factor=4, mode='bilinear', align_corners=False)
        # exit()

        bilinear_SSIM_u.append(ssimLoss(x8,hr,0,i))
        bilinear_SSIM_v.append(ssimLoss(x8,hr,1,i))

        bilinear_PSNR_u.append(psnrLoss(x8,hr,0,i))
        bilinear_PSNR_v.append(psnrLoss(x8,hr,1,i))

        bilinear_lpips_u.append(lpipsLoss(x8,hr,0,i))
        bilinear_lpips_v.append(lpipsLoss(x8,hr,1,i))

        bilinear_FSIM_u.append(fsimLoss(x8,hr,0,i))
        bilinear_FSIM_v.append(fsimLoss(x8,hr,1,i))

        bilinear_EPI_u.append(epiLoss(x8,hr,0,i))
        bilinear_EPI_v.append(epiLoss(x8,hr,1,i))

        bilinear_TKE_u.append(LossTKE(x8,hr,0,i))
        bilinear_TKE_v.append(LossTKE(x8,hr,1,i))


        x8 = x8.cpu().squeeze(0).numpy() 

        bilinear_MSE_u.append(np.mean(np.square(high[0,:,:] - x8[0,:,:])))
        bilinear_MSE_v.append(np.mean(np.square(high[1,:,:] - x8[1,:,:])))





        ###  bicubic     #############################################################################

        x8c = F.interpolate(x.unsqueeze(0), scale_factor=4, mode='bicubic', align_corners=False)

        bic_SSIM_u.append(ssimLoss(x8c,hr,0,i))
        bic_SSIM_v.append(ssimLoss(x8c,hr,1,i))   

        bic_PSNR_u.append(psnrLoss(x8c,hr,0,i))
        bic_PSNR_v.append(psnrLoss(x8c,hr,1,i))

        bic_lpips_u.append(lpipsLoss(x8c,hr,0,i))
        bic_lpips_v.append(lpipsLoss(x8c,hr,1,i))

        bic_FSIM_u.append(fsimLoss(x8c,hr,0,i))
        bic_FSIM_v.append(fsimLoss(x8c,hr,1,i))

        bic_EPI_u.append(epiLoss(x8c,hr,0,i))
        bic_EPI_v.append(epiLoss(x8c,hr,1,i))

        bic_TKE_u.append(LossTKE(x8c,hr,0,i))
        bic_TKE_v.append(LossTKE(x8c,hr,1,i))


        x8c = x8c.cpu().squeeze(0).numpy()

        bic_MSE_u.append(np.mean(np.square(high[0,:,:] - x8c[0,:,:])))
        bic_MSE_v.append(np.mean(np.square(high[1,:,:] - x8c[1,:,:])))

        if (i==day):
            #----------------------------------------------------------
            vmin1,vmax1 = 0.9*low[0,:,:].min(),1.1*low[0,:,:].max()
            vmin2,vmax2 = 0.9*low[1,:,:].min(),1.1*low[1,:,:].max()
            #----------------------------------------------------------
            plt.figure(figsize=(20, 12))
            fig, axes = plt.subplots(2, 5, figsize=(20, 12))
            axes = axes.flatten()
            #----------------------------------------------------------
            fig.text(0.02, 0.70 - 0 * 0.25, "u", va='center', ha='left', fontsize=12, fontweight='bold', rotation=90)
            fig.text(0.02, 0.70 - 1 * 0.45, "v", va='center', ha='left', fontsize=12, fontweight='bold', rotation=90)
            #----------------------------------------------------------
            axes[0].set_title('Low Resolution')
            axes[0].imshow(low[0,:,:],vmin=vmin1, vmax=vmax1)
            # axes[0].set_xlabel(f"MSE {LR_MSE_u[-1]:0.06f} | SSIM {LR_SSIM_u[-1]:0.06f}", labelpad=8)
            axes[0].set_xlabel(
                f"$\\bf{{MSE}}$: {LR_MSE_u[-1]:0.06f} \t $\\bf{{SSIM}}$: {LR_SSIM_u[-1]:0.06f}\n"
                f"$\\bf{{PSNR}}$: {LR_PSNR_u[-1]:0.06f} \t $\\bf{{LPIPS}}$: {LR_lpips_u[-1]:0.06f}\n"
                f"$\\bf{{FSIM}}$: {LR_FSIM_u[-1]:0.06f} \t $\\bf{{EPI}}$: {LR_EPI_u[-1]:0.06f}\n"
                f"$\\bf{{TKE}}$: {LR_TKE_u[-1]:0.06f}",
                labelpad=8, fontsize=12
            )
            axes[1].set_title('2D_error')
            axes[1].imshow(x8[0,:,:],vmin=vmin1, vmax=vmax1)
            # axes[1].set_xlabel(f"MSE {bilinear_MSE_u[-1]:0.06f} | SSIM {bilinear_SSIM_u[-1]:0.06f}", labelpad=8)
            axes[1].set_xlabel(
                f"$\\bf{{MSE}}$: {bilinear_MSE_u[-1]:0.06f} \t $\\bf{{SSIM}}$: {bilinear_SSIM_u[-1]:0.06f}\n"
                f"$\\bf{{PSNR}}$: {bilinear_PSNR_u[-1]:0.06f} \t $\\bf{{LPIPS}}$: {bilinear_lpips_u[-1]:0.06f}\n"
                f"$\\bf{{FSIM}}$: {bilinear_FSIM_u[-1]:0.06f} \t $\\bf{{EPI}}$: {bilinear_EPI_u[-1]:0.06f}\n"
                 f"$\\bf{{TKE}}$: {bilinear_TKE_u[-1]:0.06f}",
                labelpad=8, fontsize=12
            )
            axes[2].set_title('Output')
            axes[2].imshow(x8c[0,:,:],vmin=vmin1, vmax=vmax1)
            # axes[2].set_xlabel(f"MSE {bic_MSE_u[-1]:0.06f} | SSIM {bic_SSIM_u[-1]:0.06f}", labelpad=8)
            axes[2].set_xlabel(
                f"$\\bf{{MSE}}$: {bic_MSE_u[-1]:0.06f} \t $\\bf{{SSIM}}$: {bic_SSIM_u[-1]:0.06f}\n"
                f"$\\bf{{PSNR}}$: {bic_PSNR_u[-1]:0.06f} \t $\\bf{{LPIPS}}$: {bic_lpips_u[-1]:0.06f}\n"
                f"$\\bf{{FSIM}}$: {bic_FSIM_u[-1]:0.06f} \t $\\bf{{EPI}}$: {bic_EPI_u[-1]:0.06f}\n"
                f"$\\bf{{TKE}}$: {bic_TKE_u[-1]:0.06f}",
                labelpad=8, fontsize=12
            )

            axes[3].set_title('Out+2Derr')
            axes[3].imshow(output[0,:,:],vmin=vmin1, vmax=vmax1)
            # axes[3].set_xlabel(f"MSE {model_MSE_u[-1]:0.06f} | SSIM {model_SSIM_u[-1]:0.06f}", labelpad=8)
            axes[3].set_xlabel(
                f"$\\bf{{MSE}}$: {model_MSE_u[-1]:0.06f} \t $\\bf{{SSIM}}$: {model_SSIM_u[-1]:0.06f}\n"
                f"$\\bf{{PSNR}}$: {model_PSNR_u[-1]:0.06f} \t $\\bf{{LPIPS}}$: {model_lpips_u[-1]:0.06f}\n"
                f"$\\bf{{FSIM}}$: {model_FSIM_u[-1]:0.06f} \t $\\bf{{EPI}}$: {model_EPI_u[-1]:0.06f}\n"
                f"$\\bf{{TKE}}$: {model_TKE_u[-1]:0.06f}",
                labelpad=8, fontsize=12
            )

            axes[4].set_title('High Resolution')
            axes[4].imshow(high[0,:,:],vmin=vmin1, vmax=vmax1)

            axes[5].imshow(low[1,:,:],vmin=vmin2, vmax=vmax2)
            # axes[5].set_xlabel(f"MSE {LR_MSE_v[-1]:0.06f} | SSIM {LR_SSIM_v[-1]:0.06f}", labelpad=8)
            axes[5].set_xlabel(
                f"$\\bf{{MSE}}$: {LR_MSE_v[-1]:0.06f} \t $\\bf{{SSIM}}$: {LR_SSIM_v[-1]:0.06f}\n"
                f"$\\bf{{PSNR}}$: {LR_PSNR_v[-1]:0.06f} \t $\\bf{{LPIPS}}$: {LR_lpips_v[-1]:0.06f}\n"
                f"$\\bf{{FSIM}}$: {LR_FSIM_v[-1]:0.06f} \t $\\bf{{EPI}}$: {LR_EPI_v[-1]:0.06f}\n"
                f"$\\bf{{TKE}}$: {LR_TKE_v[-1]:0.06f}",
                labelpad=8, fontsize=12
            )

            axes[6].imshow(x8[1,:,:],vmin=vmin2, vmax=vmax2)
            # axes[6].set_xlabel(f"MSE {bilinear_MSE_v[-1]:0.06f} | SSIM {bilinear_SSIM_v[-1]:0.06f}", labelpad=8)
            axes[6].set_xlabel(
                f"$\\bf{{MSE}}$: {bilinear_MSE_v[-1]:0.06f} \t $\\bf{{SSIM}}$: {bilinear_SSIM_v[-1]:0.06f}\n"
                f"$\\bf{{PSNR}}$: {bilinear_PSNR_v[-1]:0.06f} \t $\\bf{{LPIPS}}$: {bilinear_lpips_v[-1]:0.06f}\n"
                f"$\\bf{{FSIM}}$: {bilinear_FSIM_v[-1]:0.06f} \t $\\bf{{EPI}}$: {bilinear_EPI_v[-1]:0.06f}\n"
                f"$\\bf{{TKE}}$: {bilinear_TKE_v[-1]:0.06f}",

                labelpad=8, fontsize=12
            )


            axes[7].imshow(x8c[1,:,:],vmin=vmin2, vmax=vmax2)
            # axes[7].set_xlabel(f"MSE {bic_MSE_v[-1]:0.06f} | SSIM {bic_SSIM_v[-1]:0.06f}", labelpad=8)
            axes[7].set_xlabel(
                f"$\\bf{{MSE}}$: {bic_MSE_v[-1]:0.06f} \t $\\bf{{SSIM}}$: {bic_SSIM_v[-1]:0.06f}\n"
                f"$\\bf{{PSNR}}$: {bic_PSNR_v[-1]:0.06f} \t $\\bf{{LPIPS}}$: {bic_lpips_v[-1]:0.06f}\n"
                f"$\\bf{{FSIM}}$: {bic_FSIM_v[-1]:0.06f} \t $\\bf{{EPI}}$: {bic_EPI_v[-1]:0.06f}\n"
                f"$\\bf{{TKE}}$: {bic_TKE_v[-1]:0.06f}",
                labelpad=8, fontsize=12
            )

            axes[8].imshow(output[1,:,:],vmin=vmin2, vmax=vmax2)
            # axes[8].set_xlabel(f"MSE {model_MSE_v[-1]:0.06f} | SSIM {model_SSIM_v[-1]:0.06f}", labelpad=8)
            axes[8].set_xlabel(
                f"$\\bf{{MSE}}$: {model_MSE_v[-1]:0.06f} \t $\\bf{{SSIM}}$: {model_SSIM_v[-1]:0.06f}\n"
                f"$\\bf{{PSNR}}$: {model_PSNR_v[-1]:0.06f} \t $\\bf{{LPIPS}}$: {model_lpips_v[-1]:0.06f}\n"
                f"$\\bf{{FSIM}}$: {model_FSIM_v[-1]:0.06f} \t $\\bf{{EPI}}$: {model_EPI_v[-1]:0.06f}\n"
                f"$\\bf{{TKE}}$: {model_TKE_v[-1]:0.06f}",
                labelpad=8, fontsize=12
            )

            axes[9].imshow(high[1,:,:],vmin=vmin2, vmax=vmax2)       


            # fig.suptitle(f"DSCMS :: Trained on HYCOM (2003-2006) :: Downsample HR :: Model 3->3 [u,v,Omega] :: Tested on Year {year} Record {day}", fontsize=16)
            fig.suptitle(f"MinMax :: Trained on HYCOM (2003-2006) :: Downsample HR :: $\\bf{{Trained\\,by\\,MSE}}$ :: Tested on Year {year} Record {day}", fontsize=16)



            #plt.tight_layout()
            plt.subplots_adjust(left=0.05)
            plt.savefig("PLOT(check).png",dpi=300,bbox_inches='tight')
            # plt.savefig("PLOT_CNN_105.png",dpi=300,bbox_inches='tight')
            exit()
            #print(l2,l2x)

# # LR array ###############################################################################
# LR_MSE_u = np.array(LR_MSE_u)
# LR_MSE_v = np.array(LR_MSE_v)

# LR_SSIM_u = np.array(LR_SSIM_u)
# LR_SSIM_v = np.array(LR_SSIM_v)

# LR_PSNR_u = np.array(LR_PSNR_u)
# LR_PSNR_v = np.array(LR_PSNR_v)

# LR_lpips_u = np.array(LR_lpips_u)
# LR_lpips_v = np.array(LR_lpips_v)

# LR_FSIM_u = np.array(LR_FSIM_u)
# LR_FSIM_v = np.array(LR_FSIM_v)

# LR_EPI_u = np.array(LR_EPI_u)
# LR_EPI_v = np.array(LR_EPI_v)

# LR_TKE_u = np.array(LR_TKE_u)
# LR_TKE_v = np.array(LR_TKE_v)


# # bilinear array ###############################################################################
# bilinear_MSE_u = np.array(bilinear_MSE_u)
# bilinear_MSE_v = np.array(bilinear_MSE_v)

# bilinear_SSIM_u = np.array(bilinear_SSIM_u)
# bilinear_SSIM_v = np.array(bilinear_SSIM_v)

# bilinear_PSNR_u = np.array(bilinear_PSNR_u)
# bilinear_PSNR_v = np.array(bilinear_PSNR_v)

# bilinear_lpips_u = np.array(bilinear_lpips_u)
# bilinear_lpips_v = np.array(bilinear_lpips_v)

# bilinear_FSIM_u = np.array(bilinear_FSIM_u)
# bilinear_FSIM_v = np.array(bilinear_FSIM_v)

# bilinear_EPI_u = np.array(bilinear_EPI_u)
# bilinear_EPI_v = np.array(bilinear_EPI_v)

# bilinear_TKE_u = np.array(bilinear_TKE_u)
# bilinear_TKE_v = np.array(bilinear_TKE_v)
# # bic array ###############################################################################

# bic_MSE_u = np.array(bic_MSE_u)
# bic_MSE_v = np.array(bic_MSE_v)

# bic_SSIM_u = np.array(bic_SSIM_u)
# bic_SSIM_v = np.array(bic_SSIM_v)

# bic_PSNR_u = np.array(bic_PSNR_u)
# bic_PSNR_v = np.array(bic_PSNR_v)

# bic_lpips_u = np.array(bic_lpips_u)
# bic_lpips_v = np.array(bic_lpips_v)

# bic_FSIM_u = np.array(bic_FSIM_u)
# bic_FSIM_v = np.array(bic_FSIM_v)

# bic_EPI_u = np.array(bic_EPI_u)
# bic_EPI_v = np.array(bic_EPI_v)

# bic_TKE_u = np.array(bic_TKE_u)
# bic_TKE_v = np.array(bic_TKE_v)

# # model array ###############################################################################

# model_MSE_u = np.array(model_MSE_u)
# model_MSE_v = np.array(model_MSE_v)

# model_SSIM_u = np.array(model_SSIM_u)
# model_SSIM_v = np.array(model_SSIM_v)

# model_PSNR_u = np.array(model_PSNR_u)
# model_PSNR_v = np.array(model_PSNR_v)

# model_lpips_u = np.array(model_lpips_u)
# model_lpips_v = np.array(model_lpips_v)

# model_FSIM_u = np.array(model_FSIM_u)
# model_FSIM_v = np.array(model_FSIM_v)

# model_EPI_u = np.array(model_EPI_u)
# model_EPI_v = np.array(model_EPI_v)

# model_TKE_u = np.array(model_TKE_u)
# model_TKE_v = np.array(model_TKE_v)




# print("# - MSE - #################################################")

# print("MSE LR_u = ",LR_MSE_u.mean(),LR_MSE_u.std())
# print("MSE LR_v = ",LR_MSE_v.mean(),LR_MSE_v.std())

# print("MSE bilinear_u = ",bilinear_MSE_u.mean(),bilinear_MSE_u.std())
# print("MSE bilinear_v = ",bilinear_MSE_v.mean(),bilinear_MSE_v.std())

# print("MSE bic_u = ",bic_MSE_u.mean(),bic_MSE_u.std())
# print("MSE bic_v = ",bic_MSE_v.mean(),bic_MSE_v.std())

# print("MSE Model_u = ",model_MSE_u.mean(),model_MSE_u.std())
# print("MSE Model_v = ",model_MSE_v.mean(),model_MSE_v.std())



# print("# - SSIM -  #################################################")

# print("SSIM LR_u = ",LR_SSIM_u.mean(),LR_SSIM_u.std())
# print("SSIM LR_v = ",LR_SSIM_v.mean(),LR_SSIM_v.std())

# print("SSIM bilinear_u = ",bilinear_SSIM_u.mean(),bilinear_SSIM_u.std())
# print("SSIM bilinear_v = ",bilinear_SSIM_v.mean(),bilinear_SSIM_v.std())

# print("SSIM bic_u = ",bic_SSIM_u.mean(),bic_SSIM_u.std())
# print("SSIM bic_v = ",bic_SSIM_v.mean(),bic_SSIM_v.std())

# print("SSIM Model_u = ",model_SSIM_u.mean(),model_SSIM_u.std())
# print("SSIM Model_v = ",model_SSIM_v.mean(),model_SSIM_v.std())

# print("# - PSNR - #################################################")
# print("PSNR LR_u = ",LR_PSNR_u.mean(),LR_PSNR_u.std())
# print("PSNR LR_v = ",LR_PSNR_v.mean(),LR_PSNR_v.std())

# print("PSNR bilinear_u = ",bilinear_PSNR_u.mean(),bilinear_PSNR_u.std())
# print("PSNR bilinear_v = ",bilinear_PSNR_v.mean(),bilinear_PSNR_v.std())

# print("PSNR bic_u = ",bic_PSNR_u.mean(),bic_PSNR_u.std())
# print("PSNR bic_v = ",bic_PSNR_v.mean(),bic_PSNR_v.std())

# print("PSNR model_u = ",model_PSNR_u.mean(),model_PSNR_u.std())
# print("PSNR model_v = ",model_PSNR_v.mean(),model_PSNR_v.std())



# print("# - lpips -  #################################################")
# print("lpips LR_u = ",LR_lpips_u.mean(),LR_lpips_u.std())
# print("lpips LR_v = ",LR_lpips_v.mean(),LR_lpips_v.std())

# print("lpips bilinear_u = ",bilinear_lpips_u.mean(),bilinear_lpips_u.std())
# print("lpips bilinear_v = ",bilinear_lpips_v.mean(),bilinear_lpips_v.std())

# print("lpips bic_u = ",bic_lpips_u.mean(),bic_lpips_u.std())
# print("lpips bic_v = ", bic_lpips_v.mean(),bic_lpips_v.std())

# print("lpips model_u = ",model_lpips_u.mean(),model_lpips_u.std())
# print("lpips model_v = ",model_lpips_v.mean(),model_lpips_v.std())


# print("# - FSIM - #################################################")

# print("FSIM LR_u = ",LR_FSIM_u.mean(),LR_FSIM_u.std())
# print("FSIM LR_v = ",LR_FSIM_v.mean(),LR_FSIM_v.std())

# print("FSIM bilinear_u = ",bilinear_FSIM_u.mean(),bilinear_FSIM_u.std())
# print("FSIM bilinear_v = ",bilinear_FSIM_v.mean(),bilinear_FSIM_v.std())

# print("FSIM bic_u = ",bic_FSIM_u.mean(),bic_FSIM_u.std())
# print("FSIM bic_v = ",bic_FSIM_v.mean(),bic_FSIM_v.std())

# print("FSIM model_u = ",model_FSIM_u.mean(),model_FSIM_u.std())
# print("FSIM model_v = ",model_FSIM_v.mean(),model_FSIM_v.std())



# print("# - EPI - #################################################")

# print("EPI LR_u = ",LR_EPI_u.mean(),LR_EPI_u.std())
# print("EPI LR_v = ",LR_EPI_v.mean(),LR_EPI_v.std())

# print("EPI bilinear_u = ",bilinear_EPI_u.mean(),bilinear_EPI_u.std())
# print("EPI bilinear_v = ",bilinear_EPI_v.mean(),bilinear_EPI_v.std())

# print("EPI bic_u = ",bic_EPI_u.mean(),bic_EPI_u.std())
# print("EPI bic_v = ",bic_EPI_v.mean(),bic_EPI_v.std())

# print("EPI model_u = ",model_EPI_u.mean(),model_EPI_u.std())
# print("EPI model_v = ",model_EPI_v.mean(),model_EPI_v.std())


# print("# - EPI - #################################################")

# print("TKE LR_u = ",LR_TKE_u.mean(),LR_TKE_u.std())
# print("TKE LR_v = ",LR_TKE_v.mean(),LR_TKE_v.std())

# print("TKE bilinear_u = ",bilinear_TKE_u.mean(),bilinear_TKE_u.std())
# print("TKE bilinear_v = ",bilinear_TKE_v.mean(),bilinear_TKE_v.std())

# print("TKE bic_u = ",bic_TKE_u.mean(),bic_TKE_u.std())
# print("TKE bic_v = ",bic_TKE_v.mean(),bic_TKE_v.std())

# print("TKE model_u = ",model_TKE_u.mean(),model_TKE_u.std())
# print("TKE model_v = ",model_TKE_v.mean(),model_TKE_v.std())





# ### HISTOGRAMS FOR PRESENTATION  #################################################################

# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# #==================================================================================================
# loss_data_u = {
#     "MSE": (LR_MSE_u, model_MSE_u),
#     "SSIM": (LR_SSIM_u, model_SSIM_u),
#     "PSNR": (LR_PSNR_u, model_PSNR_u),
#     "lpips": (LR_lpips_u, model_lpips_u),
#     "FSIM": (LR_FSIM_u, model_FSIM_u),
#     "EPI": (LR_EPI_u, model_EPI_u),
#     "TKE": (LR_TKE_u, model_TKE_u),
# }

# for i, (data1_u, data4_u) in loss_data_u.items():

#     # Plot histograms for LR and Model
#     plt.figure(figsize=(10, 6))
#     sns.histplot(data1_u, label=f"LR (μ={np.mean(data1_u):.5f}, σ={np.std(data1_u):.5f})", color="blue", kde=True, bins=30, alpha=0.5)
#     sns.histplot(data4_u, label=f"Model (μ={np.mean(data4_u):.5f}, σ={np.std(data4_u):.5f})", color="gray", kde=True, bins=30, alpha=0.5)

#     # Customize the plot
#     plt.xlabel('Value')
#     plt.ylabel('Density')
#     plt.title(f'{i} Histogram_u')
#     plt.legend()
#     plt.grid(True)

#     # Save the figure
#     plt.savefig(f"Histograms/His/{i}-histogram_u.png", dpi=300)

#     # Close the figure to free memory
#     plt.close()

# #==================================================================================================
# loss_data_v = {
#     "MSE": (LR_MSE_v, model_MSE_v),
#     "SSIM": (LR_SSIM_v, model_SSIM_v),
#     "PSNR": (LR_PSNR_v, model_PSNR_v),
#     "lpips": (LR_lpips_v, model_lpips_v),
#     "FSIM": (LR_FSIM_v, model_FSIM_v),
#     "EPI": (LR_EPI_v, model_EPI_v),
#     "TKE": (LR_TKE_v, model_TKE_v),
# }

# for i, (data1_v, data4_v) in loss_data_v.items():

#     # Plot histograms for LR and Model
#     plt.figure(figsize=(10, 6))
#     sns.histplot(data1_v, label=f"LR (μ={np.mean(data1_v):.5f}, σ={np.std(data1_v):.5f})", color="blue", kde=True, bins=30, alpha=0.5)
#     sns.histplot(data4_v, label=f"Model (μ={np.mean(data4_v):.5f}, σ={np.std(data4_v):.5f})", color="gray", kde=True, bins=30, alpha=0.5)

#     # Customize the plot
#     plt.xlabel('Value')
#     plt.ylabel('Density')
#     plt.title(f'{i} Histogram_v')
#     plt.legend()
#     plt.grid(True)

#     # Save the figure
#     plt.savefig(f"Histograms/His/{i}-histogram_v.png", dpi=300)

#     # Close the figure to free memory
#     plt.close()


# exit()



# #   Histograms ===================================================================================

# import seaborn as sns
# from scipy.stats import norm
# #==================================================================================================
# loss_data_u = {
#     "MSE": (LR_MSE_u, bilinear_MSE_u, bic_MSE_u, model_MSE_u),
#     "SSIM": (LR_SSIM_u, bilinear_SSIM_u, bic_SSIM_u, model_SSIM_u),
#     "PSNR": (LR_PSNR_u, bilinear_PSNR_u, bic_PSNR_u, model_PSNR_u),
#     "lpips": (LR_lpips_u, bilinear_lpips_u, bic_lpips_u, model_lpips_u),
#     "FSIM": (LR_FSIM_u, bilinear_FSIM_u, bic_FSIM_u, model_FSIM_u),
#     "EPI": (LR_EPI_u, bilinear_EPI_u, bic_EPI_u, model_EPI_u),
#     "TKE": (LR_TKE_u, bilinear_TKE_u, bic_TKE_u, model_TKE_u),

# }

# for i, (data1_u, data2_u, data3_u, data4_u) in loss_data_u.items():

#     # Create a range for the x-axis (using only data4_u)
#     x = np.linspace(
#         min(data4_u) - 1, 
#         max(data4_u) + 1, 
#         300
#     )

#     # Plot histogram for data4_u only
#     plt.figure(figsize=(10, 6))
#     sns.histplot(data4_u, label=f"Model (μ={np.mean(data4_u):.5f}, σ={np.std(data4_u):.5f})", color="gray", kde=True, bins=30, alpha=0.5)

#     # Customize the plot
#     plt.xlabel('Value')
#     plt.ylabel('Density')
#     plt.title(f'{i} Histogram_u')
#     plt.legend()
#     plt.grid(True)

#     # Save with dynamic filename
#     plt.savefig("Histograms/His/"f"{i}-histogram_u.png", dpi=300)
   
   
#     # Close the figure to free memory
#     plt.close()


# #========================================================================================================================
# loss_data_v = {
#     "MSE": (LR_MSE_v, bilinear_MSE_v, bic_MSE_v, model_MSE_v),
#     "SSIM": (LR_SSIM_v, bilinear_SSIM_v, bic_SSIM_v, model_SSIM_v),
#     "PSNR": (LR_PSNR_v, bilinear_PSNR_v, bic_PSNR_v, model_PSNR_v),
#     "lpips": (LR_lpips_v, bilinear_lpips_v, bic_lpips_v, model_lpips_v),
#     "FSIM": (LR_FSIM_v, bilinear_FSIM_v, bic_FSIM_v, model_FSIM_v),
#     "EPI": (LR_EPI_v, bilinear_EPI_v, bic_EPI_v, model_EPI_v),
#     "TKE": (LR_TKE_v, bilinear_TKE_v, bic_TKE_v, model_TKE_v),

# }

# for i, (data1_v, data2_v, data3_v, data4_v) in loss_data_v.items():

#     # Create a range for the x-axis (using only data4_v)
#     x = np.linspace(
#         min(data4_v) - 1, 
#         max(data4_v) + 1, 
#         300
#     )

#     # Plot histogram for data4_v only
#     plt.figure(figsize=(10, 6))
#     sns.histplot(data4_v, label=f"Model (μ={np.mean(data4_v):.5f}, σ={np.std(data4_v):.5f})", color="gray", kde=True, bins=30, alpha=0.5)

#     # Customize the plot
#     plt.xlabel('Value')
#     plt.ylabel('Density')
#     plt.title(f'{i} Histogram_v')
#     plt.legend()
#     plt.grid(True)

#     # Save with dynamic filename
#     plt.savefig("Histograms/PRUSR_His/"f"{i}-histogram_v.png", dpi=300)
 
#     # Close the figure to free memory
#     plt.close()


    

# #   Histograms ===================================================================================

# # import seaborn as sns
# # from scipy.stats import norm
# #==================================================================================================
# loss_data_u = {
#     "MSE": (LR_MSE_u, bilinear_MSE_u, bic_MSE_u, model_MSE_u),
#     "SSIM": (LR_SSIM_u, bilinear_SSIM_u, bic_SSIM_u, model_SSIM_u),
#     "PSNR": (LR_PSNR_u, bilinear_PSNR_u, bic_PSNR_u, model_PSNR_u),
#     "lpips": (LR_lpips_u, bilinear_lpips_u, bic_lpips_u, model_lpips_u),
#     "FSIM": (LR_FSIM_u, bilinear_FSIM_u, bic_FSIM_u, model_FSIM_u),
#     "EPI": (LR_EPI_u, bilinear_EPI_u, bic_EPI_u, model_EPI_u),
#     "TKE": (LR_TKE_u, bilinear_TKE_u, bic_TKE_u, model_TKE_u),

# }

# for i, (data1_u, data2_u, data3_u, data4_u) in loss_data_u.items():

#     stats = {
#         "LR": (np.mean(data1_u), np.std(data1_u)),
#         "Bilinear": (np.mean(data2_u), np.std(data2_u)),
#         "Bicubic": (np.mean(data3_u), np.std(data3_u)),
#         "Model": (np.mean(data4_u), np.std(data4_u)),
#     }

#     # Create a range for the x-axis
#     x = np.linspace(
#         min(np.concatenate([data1_u, data2_u, data3_u, data4_u])) - 1, 
#         max(np.concatenate([data1_u, data2_u, data3_u, data4_u])) + 1, 
#         300
#     )

#      # Plot histograms
#     plt.figure(figsize=(10, 6))
#     colors = ["blue", "green", "red", "gray"]
    
#     for (label, (mean, std)), color, data in zip(stats.items(), colors, [data1_u, data2_u, data3_u, data4_u]):
#         sns.histplot(data, label=f"{label} (μ={mean:.5f}, σ={std:.5f})", color=color, kde=True, bins=30, alpha=0.5)

#     # Customize the plot
#     plt.xlabel('Value')
#     plt.ylabel('Density')
#     plt.title(f'{i} Histogram_u')
#     plt.legend()
#     plt.grid(True)

#     # Save with dynamic filename
#     plt.savefig("Histograms/His/"f"{i}-histograms_u.png", dpi=300)
    
#     # Close the figure to free memory
#     plt.close()

# #========================================================================================================================
# loss_data_v = {
#     "MSE": (LR_MSE_v, bilinear_MSE_v, bic_MSE_v, model_MSE_v),
#     "SSIM": (LR_SSIM_v, bilinear_SSIM_v, bic_SSIM_v, model_SSIM_v),
#     "PSNR": (LR_PSNR_v, bilinear_PSNR_v, bic_PSNR_v, model_PSNR_v),
#     "lpips": (LR_lpips_v, bilinear_lpips_v, bic_lpips_v, model_lpips_v),
#     "FSIM": (LR_FSIM_v, bilinear_FSIM_v, bic_FSIM_v, model_FSIM_v),
#     "EPI": (LR_EPI_v, bilinear_EPI_v, bic_EPI_v, model_EPI_v),
#     "TKE": (LR_TKE_v, bilinear_TKE_v, bic_TKE_v, model_TKE_v),

# }

# for i, (data1_v, data2_v, data3_v, data4_v) in loss_data_v.items():

#     stats = {
#         "LR": (np.mean(data1_v), np.std(data1_v)),
#         "Bilinear": (np.mean(data2_v), np.std(data2_v)),
#         "Bicubic": (np.mean(data3_v), np.std(data3_v)),
#         "Model": (np.mean(data4_v), np.std(data4_v)),
#     }

#     # Create a range for the x-axis
#     x = np.linspace(
#         min(np.concatenate([data1_v, data2_v, data3_v, data4_v])) - 1, 
#         max(np.concatenate([data1_v, data2_v, data3_v, data4_v])) + 1, 
#         300
#     )

#      # Plot histograms
#     plt.figure(figsize=(10, 6))
#     colors = ["blue", "green", "red", "gray"]
    
#     for (label, (mean, std)), color, data in zip(stats.items(), colors, [data1_v, data2_v, data3_v, data4_v]):
#         sns.histplot(data, label=f"{label} (μ={mean:.5f}, σ={std:.5f})", color=color, kde=True, bins=30, alpha=0.5)

#     # Customize the plot
#     plt.xlabel('Value')
#     plt.ylabel('Density')
#     plt.title(f'{i} Histogram_v')
#     plt.legend()
#     plt.grid(True)

#     # Save with dynamic filename
#     plt.savefig("Histograms/His/"f"{i}-histograms_v.png", dpi=300)
    
#     # Close the figure to free memory
#     plt.close()