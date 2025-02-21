import express from 'express';
import { getTimetable } from '../controllers/timetablecontroller/timetablecontroller.js';
import { getAvailableTeachers } from '../controllers/timetablecontroller/timetablecontroller.js';
import { substituteTeacher } from '../controllers/timetablecontroller/timetablecontroller.js';

const router = express.Router();
router.post('/add',async (req, res) => {
    try {
        const { day, periods } = req.body;

        // Check if timetable already exists for that day
        const existingTimetable = await Timetable.findOne({ day });
        if (existingTimetable) {
            return res.status(400).json({ message: "Timetable for this day already exists" });
        }

        const newTimetable = new Timetable({ day, periods });
        await newTimetable.save();
        
        res.status(201).json({ message: "Timetable added successfully", timetable: newTimetable });
    } catch (error) {
        res.status(500).json({ message: "Server error", error: error.message });
    }
});
router.get('/getTimetable', getTimetable);
router.post('/getAvailableTeachers', getAvailableTeachers);
router.put('/substituteTeacher', substituteTeacher);

export default router;