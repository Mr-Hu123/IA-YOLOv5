#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
import time
import shutil
import numpy as np
import core.utils as utils
from tqdm import tqdm
from core.dataset import Dataset
from core.yolov5 import YOLOV5
from core.config import cfg
from core.config import args
from core.DIP import DIP
import myutils as util
from filters import *

import tensorflow
print('tensorflow.version=', tensorflow.__version__)
if tensorflow.__version__.startswith('1.'):
    import tensorflow as tf
else:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
tf.get_logger().setLevel('ERROR')

class YoloTrain(object):
    def __init__(self):

        self.anchor_per_scale       = cfg.YOLO.ANCHOR_PER_SCALE
        self.classes                = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes            = len(self.classes)
        
        self.learn_rate_init        = cfg.TRAIN.LEARN_RATE_INIT
        self.learn_rate_end         = cfg.TRAIN.LEARN_RATE_END
        self.first_stage_epochs     = cfg.TRAIN.FISRT_STAGE_EPOCHS
        self.second_stage_epochs    = cfg.TRAIN.SECOND_STAGE_EPOCHS
        self.warmup_periods         = cfg.TRAIN.WARMUP_EPOCHS
        self.initial_weight         = cfg.TRAIN.INITIAL_WEIGHT
        self.loss_file              = cfg.YOLO.LOSS_SAVE_FILE
        
        self.ckpt_path              = cfg.TRAIN.CKPT_PATH     
        self.model_path             = cfg.YOLO.MODEL_SAVE_PATH
        self.model_file             = cfg.YOLO.MODEL_FILE
        
        self.time                   = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        self.moving_ave_decay       = cfg.YOLO.MOVING_AVE_DECAY
        self.max_bbox_per_scale     = 150
        self.trainset               = Dataset('train')
        self.testset                = Dataset('test')
        self.steps_per_period       = len(self.trainset)
        
        config                      = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess                   = tf.Session(config=config)
        
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)
            
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
            with open(self.model_file, 'a') as f:
                f.write('null\n')
        
        try:
            self.model_F = tf.train.latest_checkpoint(self.ckpt_path)
        except:
            self.model_F = NULL
            print("no model here")
        

        self.log_path = ('log/')
        if os.path.exists(self.log_path):
            shutil.rmtree(self.log_path)
        os.makedirs(self.log_path)

        
        

        with tf.name_scope('define_input'):
            self.input_data   = tf.placeholder(tf.float32, [None, None, None, 3], name='input_data')
            self.defog_A   = tf.placeholder(tf.float32, [None, 3], name='defog_A')
            self.IcA   = tf.placeholder(tf.float32, [None, None, None,1], name='IcA')
            self.input_data_clean   = tf.placeholder(tf.float32, [None, None, None, 3], name='input_data')
            
            
            self.label_sbbox = tf.placeholder(dtype=tf.float32, name='label_sbbox')
            self.label_mbbox = tf.placeholder(dtype=tf.float32, name='label_mbbox')
            self.label_lbbox = tf.placeholder(dtype=tf.float32, name='label_lbbox')

            self.true_sbboxes = tf.placeholder(dtype=tf.float32, name='sbboxes')
            self.true_mbboxes = tf.placeholder(dtype=tf.float32, name='mbboxes')
            self.true_lbboxes = tf.placeholder(dtype=tf.float32, name='lbboxes')
            
            self.trainable = tf.placeholder(dtype=tf.bool, name='training')

        with tf.name_scope('define_loss'):
            self.input_data_adjust = self.input_data

            if args.ISP_FLAG: 
                self.DIP_model = DIP(self.input_data, self.trainable, self.input_data_clean, self.defog_A, self.IcA)
                self.recovery_loss = self.DIP_model.get_recovery_loss()
                self.input_data_adjust = self.DIP_model.get_adjust_image()
            

            iou_use = 1  # (0, 1, 2) ==> (giou_loss, diou_loss, ciou_loss)
            focal_use = False  # (False, True) ==> (normal, focal_loss)
            label_smoothing = 0

            self.model = YOLOV5(self.input_data_adjust, self.trainable)

            self.net_var = tf.global_variables()
            
            self.model_loss = self.model.compute_loss(
                                                    self.label_sbbox, self.label_mbbox, self.label_lbbox,
                                                    self.true_sbboxes, self.true_mbboxes, self.true_lbboxes,
                                                    iou_use, focal_use, label_smoothing)
            
            
            self.iou_loss, self.conf_loss, self.prob_loss = self.model_loss[0], self.model_loss[1], self.model_loss[2]
            
            self.loss = self.iou_loss + self.conf_loss + self.prob_loss
            

        with tf.name_scope('learn_rate'):
            self.global_step = tf.Variable(1.0, dtype=tf.float64, trainable=False, name='global_step')
            warmup_steps = tf.constant(self.warmup_periods * self.steps_per_period, dtype=tf.float64, name='warmup_steps')
            train_steps = tf.constant((self.first_stage_epochs + self.second_stage_epochs) * self.steps_per_period,
                                       dtype=tf.float64, name='train_steps')
            
            self.learn_rate = tf.cond(
                pred=self.global_step < warmup_steps,
                true_fn=lambda: self.global_step / warmup_steps * self.learn_rate_init,
                false_fn=lambda: self.learn_rate_end + 0.5 * (self.learn_rate_init - self.learn_rate_end) * \
                                (1 + tf.cos((self.global_step - warmup_steps) / (train_steps - warmup_steps) * np.pi)))
                
            global_step_update = tf.assign_add(self.global_step, 1.0)

        with tf.name_scope('define_weight_decay'):
            moving_ave = tf.train.ExponentialMovingAverage(self.moving_ave_decay).apply(tf.trainable_variables())

        with tf.name_scope('define_first_stage_train'):
            self.first_stage_trainable_var_list = []
            for var in tf.trainable_variables():
                var_name = var.op.name
                var_name_mess = str(var_name).split('/')
                
                bboxes = ['conv_sbbox', 'conv_mbbox', 'conv_lbbox']
                
                if var_name_mess[0] in bboxes:
                    self.first_stage_trainable_var_list.append(var)

            first_stage_optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss, var_list=self.first_stage_trainable_var_list)
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([first_stage_optimizer, global_step_update]):
                    with tf.control_dependencies([moving_ave]):
                        self.train_op_with_frozen_variables = tf.no_op()

        with tf.name_scope('define_second_stage_train'):
            second_stage_trainable_var_list = tf.trainable_variables()
            second_stage_optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss, var_list=second_stage_trainable_var_list)

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([second_stage_optimizer, global_step_update]):
                    with tf.control_dependencies([moving_ave]):
                        self.train_op_with_all_variables = tf.no_op()

        with tf.name_scope('loader_and_saver'):
            self.loader = tf.train.Saver(self.net_var)
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

        with tf.name_scope('summary'):
            tf.summary.scalar('learn_rate', self.learn_rate)
            tf.summary.scalar('iou_loss', self.iou_loss)
            tf.summary.scalar('conf_loss', self.conf_loss)
            tf.summary.scalar('prob_loss', self.prob_loss)
            tf.summary.scalar('total_loss', self.loss)
            
            tf.summary.scalar("recovery_loss",  self.recovery_loss)

            self.write_op = tf.summary.merge_all()
            self.summary_writer = tf.summary.FileWriter(self.log_path, graph=self.sess.graph)


    def train(self):
        self.sess.run(tf.global_variables_initializer())
        self.initial_weight = self.model_F
        try:
            print('=> Restoring weights from: %s ... ' % self.initial_weight)
            self.loader.restore(self.sess, self.initial_weight)
        except:
            print('=> %s does not exist !!!' % self.initial_weight)
            print('=> Now it starts to train IA_YOLOv5 from scratch ...')
            self.first_stage_epochs = 0

        saving = 0.0
        
        for epoch in range(1, (1 + self.first_stage_epochs + self.second_stage_epochs)):
            if epoch <= self.first_stage_epochs:
                train_op = self.train_op_with_frozen_variables
            else:
                train_op = self.train_op_with_all_variables

            pbar = tqdm(self.trainset, leave = False, desc = 'train第%d轮'%epoch)
            
            train_epoch_loss, test_epoch_loss = [], []
            iou_epoch_loss, conf_epoch_loss, prob_epoch_loss = [], [], []
            
            for train_data in pbar:
                # whether use Hybrid data training
                if args.fog_FLAG:
                    dark = np.zeros((train_data[0].shape[0], train_data[0].shape[1], train_data[0].shape[2]))
                    
                    defog_A = np.zeros((train_data[0].shape[0], train_data[0].shape[3]))
                    
                    IcA = np.zeros((train_data[0].shape[0], train_data[0].shape[1], train_data[0].shape[2]))
                    
                    if DefogFilter in cfg.filters:
                        for i in range(train_data[0].shape[0]):
                            dark_i = util.DarkChannel(train_data[0][i])
                            defog_A_i = util.AtmLight(train_data[0][i], dark_i)
                            IcA_i = util.DarkIcA(train_data[0][i], defog_A_i)
                            dark[i, ...] = dark_i
                            defog_A[i, ...] = defog_A_i
                            IcA[i, ...] = IcA_i
                    IcA = np.expand_dims(IcA, axis=-1)
                
                    _, summary, train_step_loss, train_step_recovery_loss, global_step_val, model_loss = self.sess.run(
                        [train_op, self.write_op, self.loss, self.recovery_loss, self.global_step, self.model_loss], 
                        feed_dict={self.input_data: train_data[0],
                                   self.label_sbbox: train_data[1], 
                                   self.label_mbbox: train_data[2], 
                                   self.label_lbbox: train_data[3],
                                   self.true_sbboxes: train_data[4], 
                                   self.true_mbboxes: train_data[5], 
                                   self.true_lbboxes: train_data[6], 
                                   self.trainable: True,
                                   self.input_data_clean:train_data[7],
                                   self.defog_A: defog_A,
                                   self.IcA: IcA}) 
                else:
                    _, summary, train_step_loss, global_step_val, model_loss = self.sess.run(
                        [train_op, self.write_op, self.loss, self.global_step, self.model_loss], 
                        feed_dict={self.input_data: train_data[7],
                                   self.label_sbbox: train_data[1], 
                                   self.label_mbbox: train_data[2], 
                                   self.label_lbbox: train_data[3],
                                   self.true_sbboxes: train_data[4], 
                                   self.true_mbboxes: train_data[5], 
                                   self.true_lbboxes: train_data[6], 
                                   self.trainable: True,
                                   self.input_data_clean:train_data[7],}) 

                train_epoch_loss.append(train_step_loss)
                iou_epoch_loss.append(model_loss[0])
                conf_epoch_loss.append(model_loss[1])
                prob_epoch_loss.append(model_loss[2])
                
                self.summary_writer.add_summary(summary, global_step_val)
                pbar.set_description('train loss: %.2f' %train_step_loss)
            
            qbar = tqdm(self.testset, leave = False, desc = 'test第%d轮'%epoch)
            
            for test_data in qbar:
                if args.fog_FLAG:
                    
                    dark = np.zeros((test_data[0].shape[0], test_data[0].shape[1], test_data[0].shape[2]))
                    defog_A = np.zeros((test_data[0].shape[0], test_data[0].shape[3]))
                    IcA = np.zeros((test_data[0].shape[0], test_data[0].shape[1], test_data[0].shape[2]))
                    if DefogFilter in cfg.filters:
                        for i in range(test_data[0].shape[0]):
                            dark_i = util.DarkChannel(test_data[0][i])
                            defog_A_i = util.AtmLight(test_data[0][i], dark_i)
                            IcA_i = util.DarkIcA(test_data[0][i], defog_A_i)
                            dark[i, ...] = dark_i
                            defog_A[i, ...] = defog_A_i
                            IcA[i, ...] = IcA_i


                    IcA = np.expand_dims(IcA, axis=-1)
                    
                    test_step_loss = self.sess.run(self.loss, 
                        feed_dict={self.input_data: test_data[0],
                                   self.label_sbbox: test_data[1], 
                                   self.label_mbbox: test_data[2], 
                                   self.label_lbbox: test_data[3],
                                   self.true_sbboxes: test_data[4], 
                                   self.true_mbboxes: test_data[5], 
                                   self.true_lbboxes: test_data[6], 
                                   self.trainable: False,
                                   self.input_data_clean:test_data[7],
                                   self.defog_A: defog_A,
                                   self.IcA: IcA}) 
                else:
                    test_step_loss = self.sess.run(self.loss, 
                        feed_dict={self.input_data: test_data[7],
                                   self.label_sbbox: test_data[1], 
                                   self.label_mbbox: test_data[2], 
                                   self.label_lbbox: test_data[3],
                                   self.true_sbboxes: test_data[4], 
                                   self.true_mbboxes: test_data[5], 
                                   self.true_lbboxes: test_data[6], 
                                   self.trainable: False,
                                   self.input_data_clean:test_data[7],}) 
                    
                test_epoch_loss.append(test_step_loss)
                qbar.set_description('test loss: %.2f' %test_step_loss)
                
            train_epoch_loss, test_epoch_loss = np.mean(train_epoch_loss), np.mean(test_epoch_loss)
            
            iou_epoch_loss, conf_epoch_loss, prob_epoch_loss = \
                np.mean(iou_epoch_loss), np.mean(conf_epoch_loss), np.mean(prob_epoch_loss)
            
            
            
            
            file_name = 'IA_YOLOv5_test-loss=%.4f.ckpt' % test_epoch_loss
            ckpt_file = os.path.join(self.ckpt_path, file_name).replace('\\','/')
            
            log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            
            #
            if saving == 0.0:
                saving = train_epoch_loss
                print('=> Epoch: %2d Time: %s Train loss: %.2f' % (epoch, log_time, train_epoch_loss))
            
            elif saving > train_epoch_loss:
                print('=> Epoch: %2d Time: %s Train loss: %.2f Test loss: %.2f Saving %s ...' % 
                     (epoch, log_time, train_epoch_loss, test_epoch_loss, ckpt_file))
                self.saver.save(self.sess, ckpt_file, global_step=epoch)
                saving = train_epoch_loss
            else:
                print('=> Epoch: %2d Time: %s Train loss: %.2f' % (epoch, log_time, train_epoch_loss))


            with open(self.model_file, 'a') as f:
                f.write(ckpt_file+'\n')

            loss_save_data = str(iou_epoch_loss) + ' ' + str(conf_epoch_loss) + ' ' + str(prob_epoch_loss) \
                            + ' ' + str(epoch) + '\n'
            
            with open(self.loss_file, 'a') as f:
                f.write(loss_save_data)


if __name__ == '__main__':
    gpu_id = 0 #argv[1]
    print('train gpu_id=%s, net_type=YOLOv5' % gpu_id)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    YoloTrain().train()
