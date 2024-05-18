install.packages("warbleR")
install.packages("corrplot")
library(warbleR)
library(corrplot)


Phae.ad =read.csv("C:/Users/a0938/OneDrive/文件/sound.csv")  


Phae.ad <- autodetec(bp = c(2, 9), threshold = 20,
                     mindur = 0.09, maxdur = 0.22, ssmooth = 900, ls = TRUE,
                     res = 100, flim= c(1, 12), wl = 300, set =TRUE, sxrow =
                       6, rows = 15)

# Filter selections by signal to noise ratio
Phae.snr <- signal_2_noise(X = Phae.ad, mar = 0.04)
# Filter 5 selections from each recording
Phae.hisnr <- Phae.snr[ave(-Phae.snr$SNR, Phae.snr$sound.files, FUN = rank) <= 199,]


# Extract dominant frequency as a time series using dynamic time warping
tsLBH <- dfDTW(Phae.hisnr, length.out = 30, flim = c(1, 12), picsize = 2, res = 100,
               bp = c(2, 9))

# Filter a single selection per recording for following analyses
params <- specan(Phae.hisnr,bp=c(1,11),threshold=15)

#存檔
write.csv(params,"C:/sound/小腿肌力測試 - 複製.csv", row.names = TRUE)

# Perform spectrogram cross-correlation to compare acoustic similarity between
# filtered signals
xcor <- xcorr(X = Phae.hisnr,  wl =512, ovlp=90, dens=0.9, bp= c(20,20000), wn='hanning', 
              cor.method = "pearson", parallel = 1,  path = NULL)
# Visualize cross-correlation results
corrplot.mixed(xcor, lower.col = "black", number.cex = .7)

# Load coordinated singing data contained in package
data(sim.coor.sing)
# Visualize overlap between individuals
coor.graph(X = sim.coor.sing, ovlp = TRUE, only.coor = FALSE, xl = 2, res = 80)
# Perform randomization test to evaluate overlapping or alternating patterns
coor.test(sim.coor.sing, iterations = 100, less.than.chance = TRUE)