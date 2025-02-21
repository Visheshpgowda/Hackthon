import React, { useState, useEffect } from "react";
import axios from "axios";
import "./TimeTable.css";
import { motion } from "framer-motion";

const HMManageTimeTable = () => {
  const [groupedTimetable, setGroupedTimetable] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState(null);

  useEffect(() => {
    const fetchTimetable = async () => {
      try {
        const res = await axios.get("http://localhost:8080/api/v1/timetable/getTimetable");
        console.log("Timetable API response:", res.data);

        if (!Array.isArray(res.data)) {
          console.error("Unexpected timetable data format:", res.data);
          setError("Invalid data format received from server");
          setLoading(false);
          return;
        }

        // Group timetable data by class and day
        const groupedData = {};
        res.data.forEach((entry) => {
          const { className, day, periods } = entry;
          if (!groupedData[className]) {
            groupedData[className] = {};
          }
          groupedData[className][day] = periods;
        });

        // Ensure Saturday is reserved for sports/activities
        Object.keys(groupedData).forEach(className => {
          if (!groupedData[className]["Saturday"]) {
            groupedData[className]["Saturday"] = periods.map(periodNumber => ({
              periodNumber,
              subject: periodNumber % 2 === 0 ? "Sports" : "Rotational Activities",
              teacher: { name: "Coach Team" }
            }));
          } else {
            // Override existing Saturday data
            groupedData[className]["Saturday"] = groupedData[className]["Saturday"].map(period => ({
              ...period,
              subject: period.periodNumber % 2 === 0 ? "Sports" : "Rotational Activities",
              teacher: { name: "Coach Team" }
            }));
          }
        });

        setGroupedTimetable(groupedData);
        setActiveTab(Object.keys(groupedData)[0] || null);
        setLoading(false);
      } catch (err) {
        console.error("Error fetching timetable:", err);
        setError("Failed to load timetable data");
        setLoading(false);
      }
    };

    fetchTimetable();
  }, []);

  const days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
  const periods = [1, 2, 3, 4, 5, 6];

  // Cell background colors based on subjects
  const getSubjectColor = (subject, day) => {
    if (day === "Saturday") return "#f8d7da"; // Light red for Saturday activities
    
    const colorMap = {
      "Mathematics": "#d1ecf1", // Light blue
      "Science": "#d4edda", // Light green
      "English": "#fff3cd", // Light yellow
      "History": "#f8d7da", // Light red
      "Geography": "#e2e3e5", // Light gray
      "Computer": "#cce5ff", // Blue
      "Art": "#f5c6cb", // Pink
      "Music": "#c3e6cb", // Green
      "Physical Education": "#ffeeba", // Yellow
    };
    
    return colorMap[subject] || "#f8f9fa"; // Default light background
  };

  // Animation variants
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: { 
      opacity: 1,
      transition: { 
        staggerChildren: 0.1,
        delayChildren: 0.2
      }
    }
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: { 
      y: 0, 
      opacity: 1,
      transition: { type: "spring", stiffness: 100 }
    }
  };

  const headerVariants = {
    hidden: { y: -50, opacity: 0 },
    visible: { 
      y: 0, 
      opacity: 1, 
      transition: { 
        type: "spring", 
        stiffness: 100, 
        damping: 15 
      }
    }
  };

  return (
    <div className="timetable-management-container">
      <motion.div 
        className="timetable-header"
        initial="hidden"
        animate="visible"
        variants={headerVariants}
      >
        <h1>School Timetable Management</h1>
        <p>View and manage class schedules</p>
      </motion.div>

      {loading && (
        <motion.div 
          className="loading-container"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
        >
          <div className="loading-spinner"></div>
          <p>Loading timetables...</p>
        </motion.div>
      )}
      
      {error && (
        <motion.div 
          className="error-container"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="error-icon">⚠️</div>
          <p>{error}</p>
        </motion.div>
      )}

      {!loading && !error && (
        <motion.div 
          className="timetable-content"
          initial="hidden"
          animate="visible"
          variants={containerVariants}
        >
          {/* Class Tabs */}
          <motion.div className="class-tabs" variants={itemVariants}>
            {Object.keys(groupedTimetable).map((className) => (
              <motion.button
                key={className}
                className={`tab-button ${activeTab === className ? 'active' : ''}`}
                onClick={() => setActiveTab(className)}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                Class {className}
              </motion.button>
            ))}
          </motion.div>

          {/* Active Class Timetable */}
          {activeTab && (
            <motion.div 
              className="active-timetable"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.3 }}
            >
              <h2>Class {activeTab} Weekly Schedule</h2>
              <div className="timetable-wrapper">
                <table className="timetable">
                  <thead>
                    <tr>
                      <th className="corner-header">Period</th>
                      {days.map((day) => (
                        <th 
                          key={day} 
                          className={`day-header ${day === "Saturday" ? "saturday-header" : ""}`}
                        >
                          {day}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {periods.map((period) => (
                      <tr key={period}>
                        <td className="period-number">Period {period}</td>
                        {days.map((day) => {
                          const periodData = groupedTimetable[activeTab][day]?.find(
                            (p) => p.periodNumber === period
                          );
                          const isSpecialDay = day === "Saturday";
                          
                          return (
                            <motion.td
                              key={`${activeTab}-${day}-${period}`}
                              className={`timetable-cell ${isSpecialDay ? "saturday-cell" : ""}`}
                              style={{ 
                                backgroundColor: getSubjectColor(periodData?.subject, day)
                              }}
                              whileHover={{ 
                                scale: 1.05, 
                                boxShadow: "0 5px 15px rgba(0,0,0,0.1)" 
                              }}
                              transition={{ duration: 0.2 }}
                            >
                              {periodData ? (
                                <div className="cell-content">
                                  <div className="subject">{periodData.subject}</div>
                                  <div className="teacher-name">{periodData.teacher.name}</div>
                                </div>
                              ) : (
                                <div className="free-text">Free Period</div>
                              )}
                            </motion.td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </motion.div>
          )}
        </motion.div>
      )}
    </div>
  );
};

export default HMManageTimeTable;