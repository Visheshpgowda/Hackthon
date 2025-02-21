import { FacultyAttendance } from "../../models/Facultyattd.models.js";
import { User } from "../../models/user.models.js";

export const getfaculty= async(req, res) => {
    try {
        const faculty = await User.find({ role: 'Teacher' });
        res.status(200).json({ faculty });
    } catch (error) {
        res.status(500).json({ message: error.message });
    }
}
export const AddfacultyAttendance = async(req, res) => {
    try {
        const { facultyId, date, status } = req.body;
        if(!facultyId || !date || !status) {
            return res.status(400).json({ message: "Faculty ID, date, and status are required" });
        }   
        const faculty = await FacultyAttendance.findOne({ facultyId, date });
        if(faculty) {
            const updatedFaculty = await FacultyAttendance.findOneAndUpdate({ facultyId, date }, { status
            }, { new: true });
            return res.status(200).json({ faculty: updatedFaculty });
        }   
        
        const facultyAttendance = new FacultyAttendance({ facultyId, date, status });
        await facultyAttendance.save();
        res.status(201).json({ facultyAttendance });
    } catch (error) {
        res.status(409).json({ message: error.message });
    }
}

export const getAllFacultyAttendance = async(req, res) => {
    try {
        // Get all attendance records
        const allAttendanceRecords = await FacultyAttendance.find();

        // Group records by faculty ID
        const facultyStats = {};

        // Process each attendance record
        allAttendanceRecords.forEach(record => {
            if (!facultyStats[record.facultyId]) {
                facultyStats[record.facultyId] = {
                    facultyId: record.facultyId,
                    totalDays: 0,
                    presentDays: 0,
                    absentDays: 0,
                    leaveDays: 0,
                    attendancePercentage: 0
                };
            }

            const stats = facultyStats[record.facultyId];
            stats.totalDays++;

            switch (record.status) {
                case 'present':
                    stats.presentDays++;
                    break;
                case 'absent':
                    stats.absentDays++;
                    break;
            }
        });

        // Calculate percentages and convert to array
        const facultyStatsArray = Object.values(facultyStats).map(stats => ({
            ...stats,
            attendancePercentage: stats.totalDays > 0 ?
                ((stats.presentDays / stats.totalDays) * 100).toFixed(2) : 0
        }));

        // Optional: Sort by attendance percentage in descending order
        facultyStatsArray.sort((a, b) => b.attendancePercentage - a.attendancePercentage);

        res.status(200).json({
            totalFaculty: facultyStatsArray.length,
            stats: facultyStatsArray
        });
    } catch (error) {
        res.status(500).json({
            message: "Error fetching faculty attendance statistics",
            error: error.message
        });
    }
};

export const getMonthlyAttendanceReport = async(req, res) => {
    try {
        const { facultyId, month, year } = req.body;

        if (!facultyId || !month || !year) {
            return res.status(400).json({
                message: "Faculty ID, month, and year are required"
            });
        }

        // Create date range for the specified month
        const startDate = new Date(year, month - 1, 1);
        const endDate = new Date(year, month, 0);

        const attendanceRecords = await FacultyAttendance.find({
            facultyId,
            date: {
                $gte: startDate,
                $lte: endDate
            }
        }).sort({ date: 1 });

        // Calculate statistics
        const stats = {
            totalDays: attendanceRecords.length,
            presentDays: attendanceRecords.filter(record => record.status === 'present').length,
            absentDays: attendanceRecords.filter(record => record.status === 'absent').length,
            attendancePercentage: (
                (attendanceRecords.filter(record => record.status === 'present').length /
                    attendanceRecords.length) * 100
            ).toFixed(2)
        };

        res.status(200).json({
            month: startDate.toLocaleString('default', { month: 'long' }),
            year: year,
            stats: stats,
            data: attendanceRecords
        });
    } catch (error) {
        res.status(500).json({
            message: "Error fetching monthly attendance",
            error: error.message
        });
    }
};

export const getYearlyAttendanceReport = async(req, res) => {
    try {
        const { facultyId, year } = req.body;

        if (!facultyId || !year) {
            return res.status(400).json({
                message: "Faculty ID and year are required"
            });
        }

        // Create date range for the entire year
        const startDate = new Date(year, 0, 1);
        const endDate = new Date(year, 11, 31);

        const attendanceRecords = await FacultyAttendance.find({
            facultyId,
            date: {
                $gte: startDate,
                $lte: endDate
            }
        }).sort({ date: 1 });

        // Calculate monthly statistics
        const monthlyStats = {};
        const months = [
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ];

        months.forEach((month, index) => {
            const monthRecords = attendanceRecords.filter(record => {
                const recordDate = new Date(record.date);
                return recordDate.getMonth() === index;
            });

            monthlyStats[month] = {
                totalClasses: monthRecords.length,
                presentDays: monthRecords.filter(record => record.status === 'present').length,
                absentDays: monthRecords.filter(record => record.status === 'absent').length,
                attendancePercentage: monthRecords.length ?
                    ((monthRecords.filter(record => record.status === 'present').length /
                        monthRecords.length) * 100).toFixed(2) : 0
            };
        });

        // Calculate yearly totals
        const yearlyTotals = {
            totalClasses: attendanceRecords.length,
            totalPresent: attendanceRecords.filter(record => record.status === 'present').length,
            totalAbsent: attendanceRecords.filter(record => record.status === 'absent').length,
            yearlyAttendancePercentage: attendanceRecords.length ?
                ((attendanceRecords.filter(record => record.status === 'present').length /
                    attendanceRecords.length) * 100).toFixed(2) : 0
        };

        res.status(200).json({
            year: year,
            yearlyTotals: yearlyTotals,
            monthlyStats: monthlyStats,
            data: attendanceRecords
        });
    } catch (error) {
        res.status(500).json({
            message: "Error fetching yearly attendance",
            error: error.message
        });
    }
};