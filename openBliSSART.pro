TEMPLATE = subdirs

SUBDIRS += \
        sub_core

# Build tests on every platform

SUBDIRS += \
        sub_LibFeature \
        sub_LibFramework \
        sub_LibICA \
        sub_LibLinAlg

##SUBDIRS += \
##	checker \
##	sub_server \
##        sub_convert \
##	sub_sv

sub_core.file = sub_core.pro

sub_LibFeature.file = ./src/LibFeature/sub_LibFeature.pro
sub_LibFramework.file = ./src/LibFramework/sub_LibFramework.pro
sub_LibICA.file = ./src/LibICA/sub_LibICA.pro
sub_LibLinAlg.file = ./src/LibLinAlg/sub_LibLinAlg.pro

##sub_server.file = server.pro
##sub_convert.file = convert.pro
##sub_sv.file = sv.pro

CONFIG += ordered
