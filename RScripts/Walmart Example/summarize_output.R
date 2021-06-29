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

save_fit_obj <- function(fit, fname){
  to_save <- data.frame(unlist(fit))
  write.table(to_save, file = fname, sep = "") # How do I add the name of the category??
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