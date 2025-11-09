# Digital Twin Validation & Calibration System
## 🧭 System Navigation

### Getting Started
1. **Launch**: Run `START_SYSTEM.bat` (Windows) or `python start_system.py`
2. **Open Dashboard**: Navigate to http://localhost:5000/dashboard_with_trending.html
3. **Check Status**: Verify system health in the top status bar

### Dashboard Sections

**Arena Control Panel** (Top-left)
- Connect to Arena simulation
- Load model (configure path in `system_config.json`)
- Start/Stop simulation runs
- Monitor connection status

**Validation Controls** (Center)
- Select validation algorithm (mLCSS/DTW/LCSS)
- Set similarity thresholds
- Start validation campaigns
- View real-time validation scores

**Calibration Panel** (Right)
- Monitor parameter drift
- View calibration status (idle/active/completed)
- Track particle filter progress
- Review optimized parameters

**Activity Log** (Bottom)
- Real-time system events
- Error messages and alerts
- Operation timestamps

**Trending Charts** (Main area)
- Similarity score trends over time
- Station-level performance metrics
- Calibration effectiveness visualization

### Typical Workflow
1. Configure Arena model path → Connect → Load Model
2. Select validation algorithm → Set threshold → Start Validation
3. Monitor similarity scores → System auto-calibrates on drift detection
4. Review results in trending charts and activity log
A comprehensive Flask-based system for validating and calibrating digital twin models against real-world data, with Arena simulation integration and **fixed arena control buttons**.

## 🚀 Quick Start

### Windows Users (Recommended)
```bash
START_SYSTEM.bat
```

### Advanced Users
```bash
python start_system.py
```

### Standard Start
```bash
python main.py
```

### Access the System
- **Dashboard**: http://localhost:5000/dashboard_with_trending.html
- **API Status**: http://localhost:5000/api/system/status

## 🔧 System Features

### Arena Control 
- Connect to Arena simulation software
- Load and manage simulation models
- Start/stop simulations with robust error handling
- Real-time status monitoring with health checks

### Validation Algorithms
- **mLCSS**: Modified Longest Common Subsequence (threshold: 0.90)
- **DTW**: Dynamic Time Warping with TIC metric (threshold: 0.95)
- **LCSS**: Longest Common Subsequence for events (threshold: 0.85)

### Calibration Engine
- Bootstrap Particle Filter for parameter optimization
- Multi-station calibration support (S1-S5)
- Theil's Inequality Coefficient (TIC) optimization

## ⚙️ Configuration

Edit `config/system_config.json`:
```json
{
    "system_id": "G1-5S-PL",
    "arena_config": {
        "model_path": "C:\\path\\to\\your\\model.doe",
        "output_file": "C:\\path\\to\\log.txt"
    },
    "validation_config": {
        "thresholds": {
            "lcss": 0.85,
            "mlcss": 0.90,
            "dtw": 0.95
        }
    }
}
```

## 📁 Clean File Structure

```
DTDC
├── main.py                          # Main Flask application
├── dashboard_with_trending.html     # Web dashboard
├── start_system.py                  # Enhanced startup script
├── test_system.py                   # System test suite
├── START_SYSTEM.bat                 # Windows startup
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── config/
│   └── system_config.json          # System configuration
├── components/                      # Core system components
│   ├── data_manager.py             # Data processing
│   ├── validation_engine.py        # Validation algorithms
│   ├── calibration_engine.py       # Parameter calibration
│   ├── digital_model_interface.py  # Arena COM interface
│   └── system_controller.py        # Workflow orchestration
├── validation_algorithms/           # Algorithm implementations
│   ├── mLCSS_TIC.py                # mLCSS algorithm
│   ├── dtw_tic_validator.py        # DTW algorithm
│   └── LCSS.py                     # LCSS algorithm
├── utils/                          # Utility functions
├── data/                           # Data files
└── logs/                           # Log files
```

## 🧪 Testing

Verify all fixes work correctly:
```bash
python test_system.py
```

## 📋 System Requirements

- **OS**: Windows (for Arena integration)
- **Python**: 3.8 or higher
- **Arena Software**: Installed and COM-enabled
- **Dependencies**: `pip install -r requirements.txt`

## 🛠️ Usage

1. **Configure**: Edit `config/system_config.json` with your Arena model path
2. **Start**: Run `START_SYSTEM.bat` or `python start_system.py`
3. **Access**: Open http://localhost:5000/dashboard_with_trending.html
4. **Arena Control**: Use the fixed arena control buttons in the dashboard
5. **Validate**: Start validation campaigns and monitor results

## 🔍 Troubleshooting

### Common Issues
- **Arena Connection**: Ensure Arena is installed and model path is correct
- **Port in Use**: System will show error if port 5000 is occupied
- **Dependencies**: Run `pip install -r requirements.txt` if imports fail

### Getting Help
- Check system logs in the dashboard activity log
- Run `python test_system.py` to verify system health
- Review `config/system_config.json` for configuration issues

## 📊 Data Sources

- **Station Data**: S1.csv, S2.csv, S3.csv, S5.csv
- **System KPIs**: system_kpis.csv
- **Validation Results**: validation_results.csv
- **Arena Logs**: log.txt, log.csv

## 🎯 Key Improvements

This cleaned-up version provides:
- **Streamlined file structure** - No redundant files
- **Fixed arena controls** - Reliable button operations
- **Enhanced error handling** - Better user experience
- **Simplified startup** - Clear entry points
- **Comprehensive testing** - Verify everything works

---

**Ready to use!** Start with `START_SYSTEM.bat` and access the dashboard at http://localhost:5000/dashboard_with_trending.html

