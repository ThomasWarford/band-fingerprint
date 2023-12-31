def predict(learn, item, rm_type_tfms=None, with_input=False):
    dl = learn.dls.test_dl([item], rm_type_tfms=rm_type_tfms, num_workers=0)
    inp,preds,_,dec_preds = learn.get_preds(dl=dl, with_input=True, with_decoded=True)
    return preds