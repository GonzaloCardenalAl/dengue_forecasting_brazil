# Author: Daniel Villela
# Functions to estimate Rt given temperature series
##############################################

# Function for parameter in Extrinsic Incubation Period
lambdaEIP<-function(T,v=4.3,beta0=7.9,betat=-0.21,Tbar=0) v/exp(beta0+betat*(T-Tbar))  

# Function for parameter in Intrinsic Incubation Period
lambdaIIP<-function(v=16,beta0=1.78) v/exp(beta0) 

#a1 <- 16
#a2 <- 4.3
# rate s IIP
#s1 <- lambdaIIP()
#temp <- 27
#sT <- lambdaEIP(T=temp)

# shape
#a <- c(a1, a2, 1, 1)
# converting from rate to scale parameters
#b <- c(1/s1, 1/sT, 1, 1)


# GTTemp according to Temp is previously computed
# Function requires matrix of generation time GT for legacy purposes (package R0)

#' @Cdate : vector of weekly number of new cases
#' @GT: Generation time distribution according to R0 package (for legacy purposes)
#' @GTTemp: matrix of generation time distribution 
#' as a function of temperature time series. This matrix should be obtained 
#' by using the function evalGenTimeDist.
#' @import: vector of imported cases. Not used. 
#' @q: quantiles
#' @nsim: number of simulations
est.R.Temp <- function(Cdate, GT, GTTemp, date=NULL, dataframe = NULL,
                       #Temp, 
                       import = NULL, n.t0 = NULL, t = NULL, begin = NULL, 
                       end = NULL, date.first.obs = NULL, time.step = 1, q = c(0.025, 
                                                                               0.975), 
                       correct = TRUE, nsim = 100, checked = FALSE, 
                       ...) {
  
  epid = check.incid(Cdate)#, t, date.first.obs, time.step)
  if (is.null(import)) {
    import <- rep(0, length(epid$incid))
  }
  
  if (is.null(n.t0)) {
    n.t0 = epid$incid[1]
    warning("Using initial incidence as initial number of cases.")
  }
  
  begin.nb = which(epid$t == begin)
  end.nb = which(epid$t == end)
  epid.bak <- epid
  t <- diff(c(FALSE, epid$incid == 0, FALSE), 1)
  start <- which(t == 1)
  end <- which(t == -1)
  
  #t <- 1:time
  Tmax = length(epid$incid)
  #Tmax = end
  tt <- 1:Tmax
  #  serT <- Temp[tt]
  
  ##
  
  # we use a value 9.8 just as a single point
  # But the function gives everything, including the result for 
  # this single point and the probability distribution
  #source("./sumgamma.R")
  # Now for the incubation periods
  
  a1 <- 16
  a2 <- 4.3
  # rate s IIP
  s1 <- lambdaIIP()
  temp <- 27
  sT <- lambdaEIP(T=temp)
  
  # shape
  a <- c(a1, a2, 1, 1)
  # converting from rate to scale parameters
  b <- c(1/s1, 1/sT, 1, 1)
  
  #  xx <- int_sum_gamma_T(time, a, b, Temp=serT, t=tt, max=50) 
  #  gt <- matrix(0, ncol=Tmax, nrow=Tmax)
  #  gt2 <- sapply(1:(Tmax), evaldist)
  
  #gt <- t(gt2)
  #gt <- rbind(gt, 0)
  gt2 <- GTTemp
  gt <- gt2
  gt.flip <- gt[, Tmax:1]
  
  #  for (i in 1:(Tmax-1)) {
  #    gt[i,1:length(gt2[[i]])] <- gt2[[i]]
  #  }
  #Tmax = length(epid$incid)
  
  #GT.pad = GT$GT
  #  if (length(GT.pad) < Tmax) {
  #    GT.pad <- c(GT.pad, rep(0, Tmax - length(GT.pad)))
  #  }
  
  P <- matrix(0, ncol = Tmax, nrow = Tmax)
  #  Pcorrected <- matrix(0, ncol = Tmax, nrow = Tmax)
  p <- matrix(0, ncol = Tmax, nrow = Tmax)
  multinom.simu = vector("list", Tmax)
  multinom.simu[[1]] = matrix(0, Tmax, nsim)
  if (epid$incid[1] - n.t0 > 0) {
    P[1, 1] <- (epid$incid[1] - n.t0)/(epid$incid[1] - 1)
    # Pcorrected[1, 1] <- (epid$incid[1] - n.t0)/(epid$incid[1] - 1)
    p[1, 1] <- 1    
    multinom.simu[[1]][1, ] = rmultinom(nsim, epid$incid[1] - 
                                          n.t0, p[1:1, 1])
  }
  epid.orig <- epid
  epid$incid = epid$incid + import
  for (s in 2:Tmax) {
    gtm <- gt[1:s, 1:s]
    gtm.flip <- gtm[,s:1]
    multinom.simu[[s]] = matrix(0, Tmax, nsim)
    dg <- diag(gtm.flip)
    # cat ("s", s , "diag sum ", sum(dg), "\n")
    if ((epid$incid[s] - import[s] > 0)) {
      weight.cases.for.s <- (epid$incid[1:s] - 
                               c(rep(0, s - 1), import[s])) * diag(gtm.flip)
      weight.cases.for.s <- weight.cases.for.s/sum(weight.cases.for.s)
      # I removed the 1+ import[s]  to just import[s]
      prob.cases.for.s <- 
        weight.cases.for.s * (epid$incid[s] - 
                                import[s])/(epid$incid[1:s] - 
                                              c(rep(0, s - 1), import[s]))
      prob.cases.for.s[epid$incid[1:(s - 1)] == 0] <- 0
      if (epid$incid[s] - import[s] == 1) {
        prob.cases.for.s[s] <- 0
      }
      P[1:s, s] <- prob.cases.for.s
      p[1:s, s] <- weight.cases.for.s
      #      if (correct) {
      #        Pcorrected[1:s, s] <- prob.cases.for.s/cumsum(dg)
      #      }
      multinom.simu[[s]][1:s, ] = multinom.simu[[s - 1]][1:s,] + 
        rmultinom(nsim, epid$incid[s] - import[s], p[1:s, s])
    }
    else {
      P[1:s, s] <- 0
      p[1:s, s] <- 0
      multinom.simu[[s]][1:s, ] = multinom.simu[[s - 1]][1:s, ]
    }
  }
  R.WT <- apply(P, 1, sum)
  ngt <- diag(gt.flip)
  # cat("ngt ", ngt, "\n")
  # cat("sum(ngt) ", sum(ngt), "\n")
  # cat("length(ngt) ", length(ngt), "\n")
  # cat("cumsum(ngt) ", cumsum(ngt), "\n")
  # cat("dim(gt.flip) ", dim(gt.flip), "\n")
  
  
  ngt <- ngt/sum(ngt)
  R.corrected <- R.WT/(cumsum(ngt[Tmax:1]))[Tmax:1]
  #  R.corrected <- apply(Pcorrected, 1, sum)
  if (is.na(R.corrected[length(epid$incid)])) {
    R.corrected[length(epid$incid)] <- 0
  }
  total.infected.by.time.unit.simu = multinom.simu[[length(epid$incid)]]
  R.simu <- total.infected.by.time.unit.simu/c(epid$incid)
  
  #cat("R.simu ", R.simu[Tmax,], "\n")
  R.simu.corrected <- ifelse(R.simu==0, 0, R.simu/(cumsum(ngt[Tmax:1]))[Tmax:1])
  #R.simu.corrected[end.nb,] <- ifelse(R.simu[end.nb,]==0, 0, R.simu[end.nb,]/(cumsum(ngt[1:Tmax]))[1])
  
  #cat("R.simu.corected ", R.simu.corrected[Tmax,], "\n")
  quant.simu = matrix(0, Tmax, 2)
  quant.simu.corrected = matrix(0, Tmax, 2)
  for (s in 1:Tmax) {
    if (epid$incid[s] == 0) {
      R.WT[s] <- 0
      R.simu[s] <- 0
      R.corrected[s] <- 0
      R.simu.corrected[s] <- 0
    }
    quant.simu[s, ] = quantile(R.simu[s, ], q, na.rm = TRUE)
    if (correct) {
      quant.simu.corrected[s, ] = quantile(R.simu.corrected[s, ], 
                                           q, na.rm = TRUE)
    }
  }
  conf.int = matrix(data = NA, nrow = end.nb, ncol = 2)
  colnames(conf.int) = c("lower", "upper")
  if (correct == TRUE) {
    R = R.corrected[begin.nb:end.nb]
    conf.int[begin.nb:end.nb, 1] = quant.simu.corrected[begin.nb:end.nb, 1]
    conf.int[begin.nb:end.nb, 2] = quant.simu.corrected[begin.nb:end.nb, 2]
  }
  
  if (!correct) {
    R = R.WT[begin.nb:end.nb]
    conf.int[begin.nb:end.nb, 1] = quant.simu[begin.nb:end.nb, 1]
    conf.int[begin.nb:end.nb, 2] = quant.simu[begin.nb:end.nb, 2]
  }
  names(R) = epid$t[begin.nb:end.nb]
  conf.int.orig = conf.int
  conf.int <- data.frame(na.omit(conf.int))
  #cat("conf.int length ", dim(conf.int), "end.nb ", end.nb, "\n")
  #cat("which line 1", which(is.na(conf.int.orig[,1])), "\n")
  #cat("line 1", conf.int.orig[,1], "\n")
  #cat("which line 2", which(is.na(conf.int.orig[,2])), "\n")
  rownames(conf.int) = as.character(epid$t[begin.nb:end.nb])
  pred = epid$incid
  pred[2:(length(epid$incid) + length(GT$GT))] = 0
  for (s in 1:end.nb) {
    pred[s:(s + length(GT$GT) - 1)] = pred[s:(s + length(GT$GT) - 
                                                1)] + R[s] * epid$incid[s] * GT$GT
  }
  pred = pred[1:end.nb]
  return(list(R = R, conf.int = conf.int, P = P, 
              p = p, GT = GT, epid = epid.bak, import = import, pred = pred, 
              begin = begin, begin.nb = begin.nb, end = end, end.nb = end.nb, 
              date = date,
              #                        data.name = DNAME, call = CALL, method = "Time-Dependent", 
              method.code = "TD"))
  
}


#funcoes auxiliares

# Para passar de semana epidemiologica para data
SE.tabela <- read.csv("./SE.csv")
find.date <- function(X) { 
  Y <- X
  year <- Y %/% 100; 
  sem <- Y %% 100;
  SE.tabela[(SE.tabela$Ano==year) & (SE.tabela$SE==sem),]$Inicio 
}

# Function to produce matrix of generation time distribution
#GT.max <- 10
evaldist.old <- function(x) {
  #cat("evaldist", x, "\n")
  mxx <- int_sum_gamma_T(1, a, b, Temp=serT[x:(Tmax+GT.max+1)], t=tt[x:(Tmax+GT.max+1)], max=GT.max, unitscale=7)
  #c(0, mxx$dist[(x+1):Tmax] - mxx$dist[x:(Tmax-1)])
  mxx$pdf
}

# Function to produce matrix of generation time distribution

#' @a : vector of 4 parameters for 4 gamma distributions
#' @b : vector of 4 parameters for 4 gamma distributions
#' @serT : Temperature series
#' @tt : time series
#' @GT.max : maximum number of weeks to consider for generation time
#' @return matrix with generation time distributions per week
evalGenTimeDist <- function(x, a, b, serT, tt, GT.max) {
  #cat("evaldist", x, "\n")
  mxx <- int_sum_gamma_T(1, a, b, Temp=serT[x:(Tmax+GT.max+1)], t=tt[x:(Tmax+GT.max+1)], max=GT.max, unitscale=7)
  #c(0, mxx$dist[(x+1):Tmax] - mxx$dist[x:(Tmax-1)])
  mxx$pdf
}


