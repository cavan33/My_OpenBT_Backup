# fitv=vartivity.openbt(fit)
# Do the vartivity function manually, to test things:

res=list()
res$vdraws=read.table(paste(fit$folder,"/",fit$modelname,".vdraws",sep=""))
res$vdrawsh=read.table(paste(fit$folder,"/",fit$modelname,".vdrawsh",sep=""))
res$vdraws=as.matrix(res$vdraws)
res$vdrawsh=as.matrix(res$vdrawsh)

