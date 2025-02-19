import { Marks } from "../../models/Marks.models";

export const teacherPerformance = async (req, res) => {
    try {
        const teacherId = req.params.teacher;

        // Fetch all marks for the teacher with populated student data
        const marks = await Marks.find({ teacher: teacherId })
            .populate('Student_name', 'name')
            .populate('teacher', 'name')
            .lean();

        if (marks.length === 0) {
            return res.status(404).json({ message: "No marks data found for this teacher" });
        }

        // Basic information
        const teacherName = marks[0]?.teacher?.name || "Unknown Teacher";
        const totalEntries = marks.length;
        const uniqueStudents = new Set(marks.map(m => m.Student_name?._id?.toString())).size;
        const uniqueSubjects = [...new Set(marks.map(m => m.subject))];

        // Grade distribution
        const gradeRanges = {
            'A+': { min: 90, max: 100, count: 0 },
            'A': { min: 80, max: 89.99, count: 0 },
            'B+': { min: 75, max: 79.99, count: 0 },
            'B': { min: 70, max: 74.99, count: 0 },
            'C+': { min: 65, max: 69.99, count: 0 },
            'C': { min: 60, max: 64.99, count: 0 },
            'D+': { min: 55, max: 59.99, count: 0 },
            'D': { min: 50, max: 54.99, count: 0 },
            'F': { min: 0, max: 49.99, count: 0 }
        };

        marks.forEach(mark => {
            for (const [grade, range] of Object.entries(gradeRanges)) {
                if (mark.marks >= range.min && mark.marks <= range.max) {
                    range.count++;
                    break;
                }
            }
        });

        const gradeDistribution = {};
        Object.entries(gradeRanges).forEach(([grade, range]) => {
            gradeDistribution[grade] = {
                count: range.count,
                percentage: ((range.count / totalEntries) * 100).toFixed(2)
            };
        });

        // Pass/Fail Statistics
        const passCount = marks.filter(mark => mark.marks >= 50).length;
        const failCount = marks.filter(mark => mark.marks < 50).length;
        const passRate = ((passCount / totalEntries) * 100).toFixed(2);
        const failRate = ((failCount / totalEntries) * 100).toFixed(2);

        // Performance categories
        const performanceCategories = {
            excellent: { count: 0, percentage: "0.00" },
            good: { count: 0, percentage: "0.00" },
            average: { count: 0, percentage: "0.00" },
            belowAverage: { count: 0, percentage: "0.00" },
            poor: { count: 0, percentage: "0.00" }
        };

        marks.forEach(mark => {
            if (mark.marks >= 80) performanceCategories.excellent.count++;
            else if (mark.marks >= 70) performanceCategories.good.count++;
            else if (mark.marks >= 60) performanceCategories.average.count++;
            else if (mark.marks >= 50) performanceCategories.belowAverage.count++;
            else performanceCategories.poor.count++;
        });

        Object.keys(performanceCategories).forEach(key => {
            performanceCategories[key].percentage = ((performanceCategories[key].count / totalEntries) * 100).toFixed(2);
        });

        // Subject-wise performance
        const subjectPerformance = {};
        uniqueSubjects.forEach(subject => {
            const subjectMarks = marks.filter(mark => mark.subject === subject);
            const subjectAvg = subjectMarks.reduce((sum, mark) => sum + mark.marks, 0) / subjectMarks.length;
            const highestMark = Math.max(...subjectMarks.map(mark => mark.marks));
            const lowestMark = Math.min(...subjectMarks.map(mark => mark.marks));
            const subjectPassRate = ((subjectMarks.filter(mark => mark.marks >= 50).length / subjectMarks.length) * 100).toFixed(2);

            // Standard deviation
            const mean = subjectAvg;
            const squareDiffs = subjectMarks.map(mark => Math.pow(mark.marks - mean, 2));
            const stdDev = Math.sqrt(squareDiffs.reduce((sum, diff) => sum + diff, 0) / squareDiffs.length).toFixed(2);

            subjectPerformance[subject] = {
                count: subjectMarks.length,
                averageMark: subjectAvg.toFixed(2),
                highestMark,
                lowestMark,
                passRate: subjectPassRate,
                standardDeviation: stdDev
            };
        });

        // Exam type comparison
        const examTypeComparison = ["Midterm", "Final"].reduce((acc, type) => {
            const examMarks = marks.filter(mark => mark.Exam_type === type);
            acc[type] = {
                count: examMarks.length,
                averageMark: examMarks.length ? (examMarks.reduce((sum, mark) => sum + mark.marks, 0) / examMarks.length).toFixed(2) : "0.00",
                passRate: examMarks.length ? ((examMarks.filter(mark => mark.marks >= 50).length / examMarks.length) * 100).toFixed(2) : "0.00"
            };
            return acc;
        }, {});

        // Top and bottom performers
        const studentPerformance = {};
        marks.forEach(mark => {
            const studentId = mark.Student_name?._id?.toString();
            if (!studentPerformance[studentId]) {
                studentPerformance[studentId] = {
                    studentId,
                    studentName: mark.Student_name?.name || `Student (${mark.roll_no})`,
                    rollNo: mark.roll_no,
                    totalMarks: 0,
                    examCount: 0
                };
            }

            studentPerformance[studentId].totalMarks += mark.marks;
            studentPerformance[studentId].examCount += 1;
        });

        const sortedStudents = Object.values(studentPerformance)
            .map(student => ({ ...student, averageMark: (student.totalMarks / student.examCount).toFixed(2) }))
            .sort((a, b) => b.averageMark - a.averageMark);

        const topPerformers = sortedStudents.slice(0, 5);
        const bottomPerformers = sortedStudents.slice(-5).reverse();

        // Overall statistics
        const allMarks = marks.map(mark => mark.marks);
        const avgMark = (allMarks.reduce((sum, mark) => sum + mark, 0) / allMarks.length).toFixed(2);
        const medianMark = calculateMedian(allMarks);
        const quartiles = {
            q1: calculateMedian(allMarks.slice(0, Math.floor(allMarks.length / 2))),
            q2: medianMark,
            q3: calculateMedian(allMarks.slice(Math.ceil(allMarks.length / 2))),
            iqr: (calculateMedian(allMarks.slice(Math.ceil(allMarks.length / 2))) - calculateMedian(allMarks.slice(0, Math.floor(allMarks.length / 2)))).toFixed(2)
        };

        res.status(200).json({
            teacherInfo: { teacherId, teacherName },
            overallStats: { totalEntries, uniqueStudents, uniqueSubjects: uniqueSubjects.length, avgMark, medianMark, passRate, failRate, quartiles },
            performanceCategories,
            gradeDistribution,
            subjectPerformance,
            examTypeComparison,
            topPerformers,
            bottomPerformers
        });
    } catch (error) {
        res.status(500).json({ message: "Error analyzing teacher performance", error: error.message });
    }
};

function calculateMedian(arr) {
    if (!arr.length) return "0.00";
    const sorted = arr.sort((a, b) => a - b);
    return sorted.length % 2 === 0 ? ((sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2).toFixed(2) : sorted[Math.floor(sorted.length / 2)].toFixed(2);
}
