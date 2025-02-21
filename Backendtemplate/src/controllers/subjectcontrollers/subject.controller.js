import { Subject } from '../../models/Subject.models.js';
import { User } from '../../models/user.models.js'; // Assuming you have a User model for faculty
import { FacultyAttendance } from '../../models/Facultyattd.models.js';
export const addSubject = async (req, res) => {
    try {
        const { Subject_name, teacher } = req.body;

        // Validate request
        if (!Subject_name || !teacher) {
            return res.status(400).json({ message: "Subject name and teacher ID are required" });
        }

        // Check if faculty (teacher) exists
        const facultyExists = await User.findById(teacher);
        if (!facultyExists) {
            return res.status(404).json({ message: "Faculty not found" });
        }

        // Create new subject
        const newSubject = new Subject({
            Subject_name,
            teacher: [teacher] // Storing teacher as an array
        });

        // Save subject to database
        await newSubject.save();

        res.status(201).json({ message: "Subject added successfully", subject: newSubject });
    } catch (error) {
        console.error(error);
        res.status(500).json({ message: "Server error" });
    }
};


export const getSubjects = async (req, res) => {
    try {
        // Fetch all subjects with teacher details populated
        const subjects = await Subject.find().populate('teacher', 'name').lean();


       

        res.status(200).json({ subjects: subjects });
    } catch (error) {
        console.error(error);
        res.status(500).json({ message: "Server error" });
    }
};
