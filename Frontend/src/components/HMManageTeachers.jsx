import { useEffect, useState } from "react";
import axios from "axios";

export default function FacultyManagement() {
  const [facultyList, setFacultyList] = useState([]);
  const [attendanceStats, setAttendanceStats] = useState({});
  const [loadingStates, setLoadingStates] = useState({});

  useEffect(() => {
    fetchFacultyData();
    fetchAttendanceStats();
  }, []);

  const fetchFacultyData = async () => {
    try {
      const { data } = await axios.get("http://localhost:8080/api/v1/teachers/getfaculty");
      setFacultyList(data?.faculty || []);
    } catch (error) {
      console.error("Error fetching faculty data:", error);
      setFacultyList([]);
    }
  };

  const fetchAttendanceStats = async () => {
    try {
      const { data } = await axios.get("http://localhost:8080/api/v1/teachers/getAllFacultyAttendance");
      const statsMap = {};
      data.stats.forEach(stat => {
        statsMap[stat.facultyId] = stat;
      });
      setAttendanceStats(statsMap);
    } catch (error) {
      console.error("Error fetching attendance stats:", error);
      setAttendanceStats({});
    }
  };

  const markAttendance = async (facultyId, status) => {
    setLoadingStates(prev => ({ ...prev, [facultyId]: true }));
    try {
      const date = new Date().toISOString().split("T")[0];
      await axios.post("http://localhost:8080/api/v1/teachers/addAttendance", { facultyId, date, status });
      await fetchAttendanceStats();
    } catch (error) {
      console.error("Error marking attendance:", error);
    }
    setLoadingStates(prev => ({ ...prev, [facultyId]: false }));
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-4xl font-extrabold text-gray-900 text-center mb-12 animate-fade-in">
          Faculty Dashboard
        </h1>

        <div className="mt-8">
          {facultyList.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {facultyList.map((faculty, index) => {
                const stats = attendanceStats[faculty._id] || {};
                const isLoading = loadingStates[faculty._id];
                
                return (
                  <div 
                    key={faculty._id}
                    className="transform transition-all duration-300 hover:scale-105 opacity-0 animate-slide-up"
                    style={{ animationDelay: `${index * 100}ms`, animationFillMode: 'forwards' }}
                  >
                    <div className="bg-white rounded-xl shadow-lg overflow-hidden">
                      <div className="p-6">
                        <div className="flex items-center mb-4">
                          <div className="h-12 w-12 rounded-full bg-gray-100 flex items-center justify-center">
                            <span className="text-xl font-bold text-gray-600">
                              {faculty.name.charAt(0)}
                            </span>
                          </div>
                          <div className="ml-4">
                            <h3 className="text-xl font-semibold text-gray-900">{faculty.name}</h3>
                            <p className="text-sm text-gray-500">{faculty.email}</p>
                          </div>
                        </div>

                        <div className="space-y-3">
                          <div className="flex justify-between items-center">
                            <span className="text-gray-600">Present Days</span>
                            <span className="font-semibold text-green-600">{stats.presentDays || 0}</span>
                          </div>
                          <div className="flex justify-between items-center">
                            <span className="text-gray-600">Absent Days</span>
                            <span className="font-semibold text-red-600">{stats.absentDays || 0}</span>
                          </div>
                          <div className="flex justify-between items-center">
                            <span className="text-gray-600">Attendance</span>
                            <span className="font-semibold text-blue-600">{stats.attendancePercentage || "0.00"}%</span>
                          </div>
                        </div>

                        <div className="mt-6 flex space-x-2">
                          <button
                            onClick={() => markAttendance(faculty._id, 'present')}
                            disabled={isLoading}
                            className={`flex-1 py-2 px-4 rounded-lg text-white font-medium transition-all duration-200 ${
                              isLoading ? 'bg-gray-400' : 'bg-green-500 hover:bg-green-600 transform hover:-translate-y-1'
                            }`}
                          >
                            Present
                          </button>
                          <button
                            onClick={() => markAttendance(faculty._id, 'absent')}
                            disabled={isLoading}
                            className={`flex-1 py-2 px-4 rounded-lg text-white font-medium transition-all duration-200 ${
                              isLoading ? 'bg-gray-400' : 'bg-red-500 hover:bg-red-600 transform hover:-translate-y-1'
                            }`}
                          >
                            Absent
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="text-center py-12">
              <p className="text-gray-500 text-xl">No faculty data available</p>
            </div>
          )}
        </div>
      </div>

      <style jsx>{`
        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }

        @keyframes slideUp {
          from {
            opacity: 0;
            transform: translateY(20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        .animate-fade-in {
          animation: fadeIn 0.5s ease-out;
        }

        .animate-slide-up {
          animation: slideUp 0.5s ease-out;
        }
      `}</style>
    </div>
  );
}