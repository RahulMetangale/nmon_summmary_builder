import os
import pandas as pd
import numpy as np

def process_nmon_file(file_path):
    """Process an .nmon file to extract system performance data.

    Args:
        file_path (str): Path to the .nmon file to be processed.

    Returns:
        tuple: Extracted performance metrics and system info if successful, 
               `None` otherwise on failure or if data is insufficient.
    """
    try:
        with open(file_path, 'r', encoding='ISO-8859-1') as file:
            lines = file.readlines()

        os_type = 'Linux'  # Default OS type
        lpar_data, proc_data, mem_data = [], [], []
        system_info = {}

        for line in lines:
            if line.startswith('AAA,AIX'):
                os_type = 'AIX'
            elif line.startswith('LPAR,T'):
                parts = line.strip().split(',')
                if len(parts) >= 14:  # Ensure at least 14 columns of data
                    lpar_data.append(parts)
                else:
                    print(f"Warning: LPAR line has unexpected format: {line.strip()}")
            elif line.startswith('PROC,T'):
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    proc_data.append(parts)
                else:
                    print(f"Warning: PROC line has unexpected format: {line.strip()}")
            elif line.startswith('MEM,T'):
                parts = line.strip().split(',')
                if len(parts) >= 8:
                    mem_data.append(parts)
            else:
                parse_system_info(line, system_info)

        # Handle cases where critical data is missing
        if not lpar_data:
            print(f"Warning: No LPAR data found in {file_path}")
            return None

        # Dataframe conversion and metric calculation
        lpar_df = pd.DataFrame(lpar_data)
        mem_df = pd.DataFrame(mem_data)
        
        metrics_func = calculate_aix_metrics if os_type == 'AIX' else calculate_linux_metrics
        metrics = metrics_func(lpar_df)

        # Process PROC data for run queue metrics
        rq = np.nan
        if proc_data:
            proc_df = pd.DataFrame(proc_data)
            rq = proc_df.iloc[:, 2].astype(float).quantile(0.95)

        # Calculate CPU usage 95th percentile
        cpu_usage_col_index = 2 if os_type == 'AIX' else 10  # Adjust column index based on OS
        cpu_usage = lpar_df.iloc[:, cpu_usage_col_index].astype(float)
        percentile_cpu = cpu_usage.quantile(0.95)

        # Extract system information
        memory_metrics = calculate_memory_metrics(mem_df, os_type)
        data_extracted_date = system_info.get("Date", "N/A")
        lpar_name = system_info.get('LPAR Name', 'N/A')
        machine_model = system_info.get('System Model', 'N/A')
        machine_serial = system_info.get('Machine Serial Number', 'N/A')
        processor_type = system_info.get('Processor Type', 'N/A')

        return (data_extracted_date, lpar_name, machine_model, machine_serial, 
                processor_type) + metrics + (percentile_cpu, rq) + memory_metrics

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def parse_system_info(line, system_info):
    """Parse system information from a line and update the system_info dictionary."""
    if 'AAA,date' in line:
        system_info["Date"] = line.split(",")[2]
    elif "lparname" in line:
        system_info["LPAR Name"] = line.split(",")[3].strip()
    elif "System Model:" in line:
        system_info["System Model"] = line.split(":")[-1].strip().strip('"').split(",")[1]
    elif "Machine Serial Number:" in line:
        system_info["Machine Serial Number"] = line.split(":")[-1].strip().strip('"')
    elif "Processor Type:" in line:
        system_info["Processor Type"] = line.split(":")[-1].strip().strip('"').split("_")[1]

def calculate_aix_metrics(data):
    """Calculate metrics specific to AIX systems from LPAR data."""
    try:
        count = len(data)
        total = data[2].astype(float).sum()
        max_cpu = data[2].astype(float).max()
        min_cpu = data[2].astype(float).min()
        VP = data[3].astype(float).min()
        E = data[6].astype(float).min()
        poolCPU = data[5].astype(float).min()  # 6th column
        poolIdle = data[8].astype(float).min()  # 9th column
        W = data[7].astype(float).min()
        cap = data[12].astype(float).min()  # 14th column

        avg = total / count if count > 0 else 0
        vp_e_ratio = (VP / E) * 100 if E > 0 else 0

        return (count, VP, E, vp_e_ratio, poolCPU, poolIdle, W, cap, total, min_cpu, avg, max_cpu)
    except Exception as e:
        print(f"Error calculating AIX metrics: {e}")
        return (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

def calculate_linux_metrics(data):
    """Calculate metrics specific to Linux systems from LPAR data."""
    try:
        count = len(data)
        total = data[2].astype(float).sum()
        max_cpu = data[2].astype(float).max()
        min_cpu = data[2].astype(float).min()
        VP = data[13].astype(float).sum()  # 14th column
        E = data[10].astype(float).sum()  # 11th column
        poolCPU = (data[8].astype(float) / 100).sum()  # 9th column
        poolIdle = data[21].astype(float).sum()  # 22nd column
        W = data[16].astype(float).sum()  # 17th column
        cap = data[4].astype(float).sum()  # 5th column

        avg = total / count if count > 0 else 0
        vp_e_ratio = (VP / E) * 100 if E > 0 else 0

        return (count, VP, E, vp_e_ratio, poolCPU, poolIdle, W, cap, total, min_cpu, avg, max_cpu)
    except Exception as e:
        print(f"Error calculating Linux metrics: {e}")
        return (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

def calculate_memory_metrics(mem_df, os_type):
    """Calculate memory usage metrics from memory data."""
    try:
        if os_type == 'AIX':
            used = mem_df.iloc[:, 6].astype(float) - mem_df.iloc[:, 4].astype(float)
        else:  # Linux
            used = mem_df.iloc[:, 2].astype(float) - mem_df.iloc[:, 7].astype(float)

        mem_df['used'] = used
        min_used = used.min() / 1024.0  # Convert to GB
        avg_used = used.mean() / 1024.0  # Convert to GB
        max_used = used.max() / 1024.0  # Convert to GB
        percentile_used = used.quantile(0.95) / 1024.0  # Convert to GB

        return (len(mem_df), min_used, avg_used, max_used, percentile_used)
    except Exception as e:
        print(f"Error calculating memory metrics: {e}")
        return (0, 0, 0, 0, 0)

def main():
    """Main function for processing nmon files and saving results."""
    folder_path = "./NMON_Reports/"  # Set your folder path here
    output_file = "nmon_summary.xlsx"  # Set your output file name here
    all_results = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.nmon'):
            full_path = os.path.join(folder_path, filename)
            result = process_nmon_file(full_path)
            if result:
                all_results.append((filename,) + result)

    columns = [
        "nmonfile", "Date", "LPAR Name", "System Model", "Machine Serial Number", "Processor Type",
        "Snapshots", "VP", "Entitled CPU", "VP:E", "Pool CPU", "Pool Idle", "Weight", 
        "Capped", "Total CPU", "Min CPU", "Avg CPU", "Max CPU", "95 Percentile CPU", "Run Queue 95", "Count Mem", 
        "Mim MEM Used", "Avg MEM Used", "Max MEM Used", "95 percentile GB"
    ]
    
    results_df = pd.DataFrame(all_results, columns=columns)
    results_df.to_excel(output_file, index=False)  # Save results to Excel

if __name__ == "__main__":
    main()
