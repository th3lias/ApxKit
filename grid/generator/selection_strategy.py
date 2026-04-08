from enum import Enum


class SelectionStrategy(Enum):
    LEVEL = 'level'
    CURVED = 'curved'
    HYPERBOLIC = 'hyperbolic'
    TENSOR = 'tensor'
    IPTOTAL = 'iptotal'
    IPCURVED = 'ipcurved'
    IPHYPERBOLIC = 'iphyperbolic'
    IPTENSOR = 'iptensor'
    QPTOTAL = 'qptotal'
    QPCURVED = 'qpcurved'
    QPHYPERBOLIC = 'qphyperbolic'
    QPTENSOR = 'qptensor'
