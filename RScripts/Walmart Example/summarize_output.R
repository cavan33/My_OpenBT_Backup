summarize_fitp <- function(fitp){
  print(head(fitp$mdraws[1,])); print(mean(fitp$mdraws))
  print(head(fitp$sdraws[1,])); print(mean(fitp$sdraws))
  # print(head(fitp$mmean)); print(mean(fitp$mmean)) # Will be the same as the mean values I took down above
  # print(head(fitp$smean)); print(mean(fitp$smean))
  print(head(fitp$msd)); print(mean(fitp$msd))
  print(head(fitp$ssd)); print(mean(fitp$ssd))
  print(mean(fitp$m.5)); print(mean(fitp$m.lower)); print(mean(fitp$m.upper))
  print(mean(fitp$s.5)); print(mean(fitp$s.lower)); print(mean(fitp$s.upper))
}
summarize_fitv <- function(fitv){
  print(fitv$vdraws[1:60, ]); print(mean(fitv$vdraws))
  print(fitv$vdrawsh[1:60, ]); print(mean(fitv$vdrawsh))
  print(fitv$mvdraws); print(fitv$mvdrawsh)
  print(fitv$vdraws.sd); print(fitv$vdrawsh.sd); print(fitv$vdraws.5)
  print(fitv$vdrawsh.5); print(fitv$vdraws.lower); print(fitv$vdraws.upper)
  print(fitv$vdrawsh.lower); print(fitv$vdrawsh.upper)
}
summarize_fits <- function(fits){
  print(mean(fits$vidraws)); print(mean(fits$vijdraws))
  print(mean(fits$tvidraws)); print(mean(fits$vdraws))
  print(mean(fits$sidraws)); print(mean(fits$sijdraws))
  print(mean(fits$tsidraws))
  # print(fits$msi); print(fits$msi.sd); print(fits$si.5)
  # print(fits$si.lower); print(fits$si.upper); print(fits$msij)
  # print(fits$sij.sd); print(fits$sij.5); print(fits$sij.lower)
  # print(fits$sij.upper); print(fits$mtsi); print(fits$tsi.sd)
  # print(fits$tsi.5); print(fits$tsi.lower); print(fits$tsi.upper)
}

save_fit_obj <- function(fit, fname, objtype){
  if (objtype != 'fitp'){
    to_save <- data.frame(unlist(fit))
  }
  else {
    drop <- c("mdraws", "sdraws")
    temp <- fit[!(names(fit) %in% drop)]
    to_save <- data.frame(unlist(temp))
  }
  write.table(to_save, file = paste(fname, '.txt', sep=""), sep = "")
 
  if (objtype == 'fitp'){ # (Nothing needed for objtype == 'fit')
    to_save2 <- list()
    to_save2[1] <- paste("Mean(mdraws): ", toString(mean(fit$mdraws)), "\n", sep="")
    to_save2[2] <- paste("Mean(sdraws): ", toString(mean(fit$sdraws)), "\n", sep="")
    to_save2[3] <- paste("Mean(msd): ", toString(mean(fit$msd)), "\n", sep="")
    to_save2[4] <- paste("Mean(ssd): ", toString(mean(fit$ssd)), "\n", sep="")
    to_save2[5] <- paste("Mean(m_lower): ", toString(mean(fit$m.lower)), "\n", sep="")
    to_save2[4] <- paste("Mean(m_5): ", toString(mean(fit$m.5)), "\n", sep="")
    to_save2[6] <- paste("Mean(m_upper): ", toString(mean(fit$m.upper)), "\n", sep="")
    to_save2[7] <- paste("Mean(s_lower): ", toString(mean(fit$s.lower)), "\n", sep="")
    to_save2[8] <- paste("Mean(s_5): ", toString(mean(fit$s.5)), "\n", sep="")
    to_save2[9] <- paste("Mean(s_upper): ", toString(mean(fit$s.upper)), "\n", sep="")
    write.table(to_save2, file = paste(fname, '_summary.txt', sep = ""), sep="")
  }  else if (objtype == 'fits'){
    to_save2 <- list()
    to_save2[1] <- paste("Mean(vidraws): ", toString(mean(fit$vidraws)), "\n", sep="")
    to_save2[2] <- paste("Mean(vijdraws): ", toString(mean(fit$vijdraws)), "\n", sep="")
    to_save2[3] <- paste("Mean(tvidraws): ", toString(mean(fit$tvidraws)), "\n", sep="")
    to_save2[4] <- paste("Mean(vdraws): ", toString(mean(fit$vdraws)), "\n", sep="")
    to_save2[5] <- paste("Mean(sidraws): ", toString(mean(fit$sidraws)), "\n", sep="")
    to_save2[6] <- paste("Mean(sijdraws): ", toString(mean(fit$sijdraws)), "\n", sep="")
    to_save2[7] <- paste("Mean(tsidraws): ", toString(mean(fit$tsidraws)), "\n", sep="")
    to_save2[8] <- paste("Mean(msij): ", toString(mean(fit$msij)), "\n", sep="")
    to_save2[9] <- paste("Mean(sij_sd): ", toString(mean(fit$sij.sd)), "\n", sep="")
    to_save2[10] <- paste("Mean(sij_5): ", toString(mean(fit$sij.5)), "\n", sep="")
    to_save2[11] <- paste("Mean(sij_lower): ", toString(mean(fit$sij.lower)), "\n", sep="")
    to_save2[12] <- paste("Mean(sij_upper): ", toString(mean(fit$sij.upper)), "\n", sep="")
    write.table(to_save2, file = paste(fname, '_summary.txt', sep = ""), sep="")
  }
}




save_fits_old <- function(fits, fname){
  fits2 <- fits # A copy
  # Manual removal (was giving a lot of problems in the for loop):
  fits2$vidraws <- NULL; fits2$vijdraws <- NULL; fits2$tvidraws <- NULL
  fits2$vdraws <- NULL; fits2$sidraws <- NULL; fits2$sijdraws <- NULL
  fits2$tsidraws <- NULL
  # str(fits2)
  to_save <- data.frame(unlist(fits2))
  # str(to_save)
  write.table(to_save, file = fname, sep = "")
}