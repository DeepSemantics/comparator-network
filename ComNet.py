def sim_func(query_pair):
    '''
    输入:
        query_pair:文本对，制表符隔开
    返回:
        simlarity:文本对语义相似度
    '''
    simnet_process.input_pair = query_pair
    preds_list = []
    for iter, data in enumerate(batch_data()):
        output = executor.run(program, feed=infer_feeder.feed(data), fetch_list=fetch_targets)        
        if args.task_mode == "pairwise":
            preds_list += list(map(lambda item: str(item[0]), output[1]))
        else:
            preds_list += map(lambda item: str(np.argmax(item)), output[1])

    return float(preds_list[0])
