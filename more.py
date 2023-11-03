def reval_pupulation(in_instances,condvec):
        #ret
        out_instances = []
        xreal,fakez,c1,m1,col,opt,c2,orig_real =  get_data_on_cond(condvec,sampler,batchSize,task.get_fakez(batchSize))
        xreal = torch.FloatTensor(xreal).cuda()
        if c1 is not None:
            xreal = torch.cat([xreal, c2], dim=1)
        #generates new batches of images for each generator, and then eval these sets by means (new) D
        for inst in in_instances:
            generator_trainer.set(inst.params)
            #fakez = task.get_fakez(batchSize)
            xfake = generator_trainer.gen(fakez)
            task.out_dim = xfake.shape[1]
            if c1 is not None:
                xfake = torch.cat([xfake, c1], dim=1)
            frr_score, fd_score = dis_fn(xreal, xfake, discriminator,beta)
            out_instances.append(Instance(
                frr_score, 
                fd_score,
                generator_trainer.get(),
                inst.loss_id,
                inst.pop_id,
                xfake,
                xreal,
                orig_real,
                im_parent = True
            ))
        return out_instances