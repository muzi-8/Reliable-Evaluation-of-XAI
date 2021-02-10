# 2020/8/14
# Pytorch Version 1.1
# Python  Version 3
# by muzi
# Inference
# Reference:
# /home/kuke/isprs/grad-cam-muzi-update/ErrorAnalysis.py
# /home/kuke/Desktop/muzi_saliency_map_code/IGOS/Grad-CAM/Grad-cam.py
# /home/kuke/Desktop/IGOS-master/IGOS_generate_video.py

def forward_inference(model,input_tensor):
	for par in model.parameters():
		par.requires_grad = False
	output = model(input_tensor)
	index = np.argmax(output.cpu().data.numpy())
	index_prob = torch.nn.functional.softmax(output)[0][index]
	return index,index_prob

def pre_processing(image_path):
	raw_img = cv2.imread(image_path)
	raw_img = cv2.resize(raw_img,(224,224))
	# process image #
	image_process = transforms.Compose(
		[	transforms.ToTensor(),
			transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
		])(raw_img[...,::-1].copy())
	image_process = torch.unsqueeze(image_process,dim=0)
	image_process = image_process.to(device)
	return image_process, raw_img

def blur(image_path):
	raw_img = cv2.imread(image_path)
	raw_img = cv2.resize(raw_img, (224, 224))
	img = np.float32(raw_img) / 255
	blurred_img1 = cv2.GaussianBlur(img, (11, 11), 5)
	blurred_img2 = np.float32(cv2.medianBlur(raw_img, 11)) / 255
	# mix blur
	blurred_img = (blurred_img1 + blurred_img2) / 2  # Note type is array
	# ---process blurred img---#
	blurred_img_torch = transforms.Compose(
		[
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])(blurred_img[..., ::-1].copy())
	blurred_img_torch = torch.unsqueeze(blurred_img_torch, dim=0)
	# Cpu to Gpu
	blurred_img_torch = blurred_img_torch.to(device)
	return blurred_img_torch,img

def tv_norm(input):
    img_mask = input[0,0,:]
    row_grad = torch.mean(torch.abs((img_mask[:-1,:]-img_mask[1:,:])).pow(2))
    col_grad = torch.mean(torch.abs((img_mask[:,:-1]-img_mask[:,1:])).pow(2))
    return row_grad+col_grad

'''
2020822
Function: In order to generate another lable saliency map
Read another lable
'''
# with open('/home/vis/workspace/muzi/data/evaluation data/two lable 2/Correct_another_lable/Make Lable.txt') as lines:
# 	array = lines.readlines()
# 	Imagename = []
# 	target_lable = []
# 	for item in array:
# 		item = item.strip('\n')
# 		Imagename.append(item.split(' ')[0])
# 		target_lable.append(item.split(' ')[1])
# 	dictory_target = dict(zip(Imagename, target_lable))
# print (dictory_target)
# print ('muzi')
import numpy as np
import torch
from torchvision import models
import os
import cv2
from torchvision import transforms
# use GPU
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# load model #
pretrained_vgg_net = models.vgg16(pretrained=True)
# pretrained_vgg_net = models.resnet50(pretrained=True)
pretrained_vgg_net.to(device)
pretrained_vgg_net.eval()

'''
STEP 1
For A Single Image
'''
# load image #
# load path
# input_path = './test_sample/'
# files = os.listdir(input_path)
# print (files)
# input_image_path = input_path + 'ILSVRC2012_val_00010830.JPEG'
###############################################################
'''
STEP 2
For A BUNK Image
'''
'''
Read File
To Do
'''
'''
Record Result
To Do
'''
import os
print ('muzi')
for path,dir_list,file_list in os.walk('/home/vis/workspace/muzi/data/evaluation data/experiment3_data'):
# for path, dir_list, file_list in os.walk('../../data/test_sample'):
	# 2020/8/15
	# ############################
	# Get evaluation data GT lable
	# input lable.txt
	# output dict
	# key: image_name
	# value: lable(GT)
	# ############################
	print (path)
	print (dir_list)
	print (file_list)
	print ('1')
	if path.split('/')[-1]!='data_sports':
		with open(path + '/' + 'lable.txt') as lines:
			array = lines.readlines()
			image_name_array = []
			GT_array = []
			for item in array:
				item = item.strip('\n')
				image_name_array.append(item.split(' ')[0])
				GT_array.append(item.split(' ')[1])
	# print (image_name_array)
	# print (GT_array)
			dictory = dict(zip(image_name_array, GT_array))
			print(dictory)
	for file_name in file_list:
		print(os.path.join(path,file_name))
		if file_name[:6] =='ILSVRC' and file_name[-4:]=='JPEG':
		#if file_name[:6] =='sketch' and file_name[-4:] == 'JPEG':
			input_image_path = os.path.join(path,file_name)
			# process image
			image_process, orignal_image = pre_processing(input_image_path)
			print (type(image_process))
			# blur
			blurred_img_torch,img = blur(input_image_path)
			# prediction index and the corresponding probability #
			orig_index_PV, orig_prob_PV = forward_inference(pretrained_vgg_net,image_process)

			# # Get GT label
			# GT = int(input_image_path.split('/')[-2])-1

			GT = int(dictory[file_name])
			print (GT)
			print ('predict result:')
			print (orig_index_PV)
			print (orig_prob_PV.cpu().data.numpy())
			# # python 2 code
			# with open(path+ '/' + "%s_PV_vgg.txt" % file_name, "w") as f:
			# 	print >>f,('predict result:')
			# 	if orig_index_PV == GT:
			# 		print >> f, ('1')
			# 	else:
			# 		print >> f, ('0')
			# 	print >> f,(orig_prob_PV.cpu().data.numpy())
            # python 3 code
			with open(path + '/' + "%s_PV_vgg.txt" % file_name, "w") as f:
				print (('predict result:'), file = f )
				if orig_index_PV == GT:
					print (('1'), file = f)
				else:
					print (('0'), file = f)
				print ((orig_prob_PV.cpu().data.numpy()), file = f)
				# get the ground truth label for the given category(PV category)
				f_groundtruth = open('./GroundTruth1000.txt')
				line_i = f_groundtruth.readlines()[orig_index_PV]
				f_groundtruth.close()
				print ((line_i), file = f)
				# print('line_i:', line_i)
				f.write("\n")

			# '''
			# STEP 3
			# Generate grad_cam
			# '''
			# from explainer_grad_cam import GradCam
			#
			# gradcam = GradCam(model=models.vgg16(pretrained=True), target_layer_names=["30"])
			# target = None
		    # # ############################################
		    # # # Get another lable saliency map
			# # target = int(dictory_target[file_name])
			# # ############################################
			# mask = gradcam(image_process, target)
			# #
            # # # 2020815
			# # # Test mask
			# #
			# # # print (mask)
			# # # print (type(mask))
			# # # print (mask.ndim)
			# # # print (mask.shape)
			# # # print (mask.size)
			# # # print (np.max(mask))
			# # # print (np.argmax(mask))
			# # # print (np.where(mask == np.max(mask)))
			# #
			# #
			# #
			# heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
			# heatmap = np.float32(heatmap) / 255
			# cam = heatmap + np.float32(orignal_image) / 255
			# cam = cam / np.max(cam)
			# image_name = input_image_path.split('/')[-1]
			# grad_cam_image_name = 'gradcam_' + image_name
			# grad_cam_save_dir = os.path.join(path, grad_cam_image_name)
			# cv2.imwrite(grad_cam_save_dir, np.uint8(255 * cam))
			#
			# # save mask
			# np.save(grad_cam_save_dir,mask)
			# #
			# print ('muzi')
			# print ('1')

			# # '''
			# # STEP 4
			# # Generate MASK
			# # '''
			# # # -------5. mask-------------#
			# #--initialize mask--#
			# mask_init = np.ones((28, 28), dtype=np.float32)
			# # --numpy_to_torch--#
			# from torch.autograd import Variable
			#
			# # (28,28) to (1,28,28)
			# out_put_mask_init = np.float32([mask_init])
			# out_put_torch_mask_init = torch.from_numpy(out_put_mask_init)
			# out_put_torch_mask_init = out_put_torch_mask_init.cuda()
			# out_put_torch_mask_init.unsqueeze_(
			# 	0)  # equal out_put_torch_mask_init = torch.unsqueeze(out_put_torch_mask_init,dim=0)
			# mask = Variable(out_put_torch_mask_init, requires_grad=True)
			# # -------6. upsample---------#
			# upsample = torch.nn.UpsamplingBilinear2d(size=(224, 224)).cuda()
			# # -------7. optimizer--------#
			# optimizer = torch.optim.Adam([mask], lr=0.1)
			# # -------10. Iteration--------#
			# for i in range(500):
			# 	upsampled_mask = upsample(mask)
			# 	upsampled_mask = upsampled_mask.expand(1, 3, 224, 224)
			# 	# a = image_process.mul(upsampled_mask)
			# 	# b = blurred_img_torch.mul(1-upsampled_mask)
			# 	perturbated_input = image_process.mul(upsampled_mask) + blurred_img_torch.mul(1 - upsampled_mask)
			# 	noise = np.zeros((224, 224, 3), dtype=np.float32)
			# 	cv2.randn(noise, 0, 0.2)
			#
			# 	###numpy_to_torch##########################
			# 	out_put_noise = np.transpose(noise, (2, 0, 1))
			# 	# out_put_noise.shape
			# 	out_put_torch_noise = torch.from_numpy(out_put_noise)
			# 	# out_put_torch_noise.shape
			# 	out_put_torch_noise = out_put_torch_noise.cuda()
			# 	out_put_torch_noise.unsqueeze_(0)
			# 	out_put_torch_noise.shape
			# 	from torch.autograd import Variable
			#
			# 	noise = Variable(out_put_torch_noise, requires_grad=True)
			# 	############################################
			# 	perturbated_input = perturbated_input + noise
			# 	output_perturbated_input = torch.nn.Softmax(dim=1)(pretrained_vgg_net(perturbated_input))
			# 	# loss  = 0.01 *torch.mean(torch.abs(1-mask)) +0.2*tv_norm(mask) + output_perturbated_input[0,category_out]
			#
			#
			# 	# ############################################
			# 	# # Get another lable saliency map
			# 	# orig_index_PV = int(dictory_target[file_name])
			# 	# ############################################
			# 	pre_score = output_perturbated_input[0, orig_index_PV].cpu().detach().numpy()
			# 	print ("Iter:{};Score:{:.4f}".format(i, output_perturbated_input[0, orig_index_PV]))
			# 	print ("Iter:{};Value:{:.4f}".format(i, tv_norm(mask)))
			# 	print ("Iter:{};Value:{:.4f}".format(i, torch.mean(torch.abs(1 - mask))))
			#
			# 	# loss = output_perturbated_input[0, category_out]+0.01 * torch.mean(torch.abs(1-mask)) +0.2*tv_norm(mask)
			# 	loss = output_perturbated_input[0, orig_index_PV] + 0.01 * torch.mean(
			# 		torch.abs(1 - mask)) + 0.2 * tv_norm(mask)
			# 	print ("Iter:{};Total loss:{:.4f}".format(i, loss))
			# 	print("**************************************")
			# 	# loss = output_perturbated_input[0, category_out] + 0.01 * torch.sum((1 - mask)**2) + 0.2 * tv_norm(mask)
			# 	optimizer.zero_grad()
			# 	loss.backward()
			# 	optimizer.step()
			# # -------11. save image---------#
			# upsampled_mask = upsample(mask)
			# # torch to numpy(Gpu to Cpu)-->(1,1,224,224) to (1,224,224)
			# mask_result = upsampled_mask.cpu().data.numpy()[0]
			# # (1,224,224)to(224,224,1)
			# mask_result = np.transpose(mask_result, (1, 2, 0))
			# # --normalize--#
			# mask_result = (mask_result - np.min(mask_result)) / np.max(mask_result)
			# # --demostrate important feature--#
			# mask_final = 1 - mask_result
			# #
			# # # # 2020/8/15
			# # # # search max value and search max index
			# # # print(mask_final )
			# # # print(type(mask_final ))
			# # # print(mask_final .ndim)
			# # # print(mask_final .shape)
			# # # print(mask_final .size)
			# # # print(np.max(mask_final ))
			# # # print(np.argmax(mask_final ))
			# # # print(np.where(mask_final == np.max(mask_final)))
			# # # print('muzi')
			# # #
			# # --render heatmap--#
			# heatmap = cv2.applyColorMap(np.uint8(255 * mask_final), cv2.COLORMAP_JET)
			# # -normalize heatmap-#
			# heatmap_normalize = np.float32(heatmap) / 255
			# # add heatmap and original image
			# cam = 0.7 * heatmap_normalize + img
			# cam = cam / np.max(cam)
			# image_name = input_image_path.split('/')[-1]
			# mask_image_name = 'mask_' + image_name
			# mask_save_dir = os.path.join(path, mask_image_name)
			# cv2.imwrite(mask_save_dir, np.uint8(255 * cam))
			# # save mask
			# np.save(mask_save_dir,mask_final)
			# print ('muzi')
			# print ('1')

			# '''
			# STEP 5
			# '''
			from RISE import *
			input_size = (224,224)
			mask_batch_size = 10
			explainer = RISE(pretrained_vgg_net,input_size,mask_batch_size)
			# maskspath = '../../trained mask/masks.npy'
			explainer.generate_masks(6000, 0.5, 8)
			# explainer.load_masks(maskspath)
			explainer.load_masks('./masks.npy')
			# if not os.path.isfile(maskspath):
			# 	explainer.generate_masks(6000, 0.5, 8)
			# else:
			# 	explainer.load_masks(maskspath)
			saliency = explainer(image_process)

			# ############################################
			# # Get another lable saliency map
			# orig_index_PV = int(dictory_target[file_name])
			# ############################################

			sal = saliency[orig_index_PV]
			# # Test GT
			# sal = saliency[437]
			mask = sal.cpu().numpy()
			mask = mask - np.min(mask)
			mask = mask / np.max(mask)
			#
			# # # # 2020815
			# # # # Test mask
			# # #
			# # print (mask)
			# # print (type(mask))
			# # print (mask.ndim)
			# # print (mask.shape)
			# # print (mask.size)
			# # print (np.max(mask))
			# # print (np.argmax(mask))
			# # print (np.where(mask == np.max(mask)))
			#
			heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
			heatmap = np.float32(heatmap) / 255
			cam = heatmap + np.float32(orignal_image) / 255
			cam = cam / np.max(cam)
			image_name = input_image_path.split('/')[-1]
			rise_image_name = 'rise_' + image_name
			rise_save_dir = os.path.join(path, rise_image_name)
			cv2.imwrite(rise_save_dir, np.uint8(255 * cam))

			# save mask
			np.save(rise_save_dir,mask)
			print ('muzi')


