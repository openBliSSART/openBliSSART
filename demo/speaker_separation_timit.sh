#!/bin/sh

# create training file 1
sox dr1_f_aks0_sa2.wav dr1_f_aks0_si1573.wav dr1_f_aks0_si2203.wav dr1_f_aks0_si943.wav dr1_f_aks0_sx133.wav dr1_f_aks0_sx223.wav dr1_f_aks0_sx313.wav dr1_f_aks0_sx403.wav dr1_f_aks0_sx43.wav dr1_f_aks0_nosa1.wav
# create training file 2
sox dr1_m_cpm0_sa1.wav dr1_m_cpm0_sa2.wav dr1_m_cpm0_si1194.wav dr1_m_cpm0_si1824.wav dr1_m_cpm0_sx114.wav dr1_m_cpm0_sx204.wav dr1_m_cpm0_sx24.wav dr1_m_cpm0_sx294.wav dr1_m_cpm0_sx384.wav dr1_m_cpm0_nosi564.wav
# base 1
./septool -v -c25 --export-matrices=W --normalize-matrices=W dr1_f_aks0_nosa1.wav
# base 2
./septool -v -c25 --export-matrices=W --normalize-matrices=W dr1_m_cpm0_nosi564.wav
# create mixed file
sox dr1_m_cpm0_si564.wav dr1_m_cpm0_si564_norm.wav norm -0.1
sox dr1_f_aks0_sa1.wav dr1_f_aks0_sa1_norm.wav norm -0.1
sox -m -v 1 dr1_f_aks0_sa1_norm.wav dr1_m_cpm0_si564_norm.wav dr1_f_aks0_sa1_vs_dr1_m_cpm0_si564.wav
# separate!
./septool -v --init-files=dr1_f_aks0_nosa1_W.dat,dr1_m_cpm0_nosi564_W.dat -P --export-components=1..25 --export-components=26..50 -c50 --mix dr1_f_aks0_sa1_vs_dr1_m_cpm0_si564.wav
