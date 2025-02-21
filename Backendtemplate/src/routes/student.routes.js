import express from 'express';
import { addMarks } from '../controllers/studentControllers/marks.controllers.js';
import { addStudent } from '../controllers/studentControllers/addStudent.controllers.js';
import { getStudentListByClass } from '../controllers/studentControllers/studentListByClass.controllers.js';
import { verifyJWT } from '../middlewares/auth.middlewares.js';
import { getEnrolledSubjects } from '../controllers/studentControllers/studentListByClass.controllers.js';
import Subject from '../models/Subject.models.js'; 
const router = express.Router();


router.post('/addStudent', addStudent); 
// router.post('/addMarks', addMarks);  
router.route('/addMarks').post(verifyJWT, addMarks); 
router.post('/getStudentListByClass', getStudentListByClass);
router.get('/getEnrolledSubjects/:studentId', getEnrolledSubjects); 

router.get('/subjects', async (req, res) => {
    try {
        console.log("CAME FOR TH$E SUBJEVTS"); 
        const subjects = await Subject.find(); 
        
        res.json(subjects);
    } catch (error) {
        res.status(500).json({ message: error.message });
    }
});
export default router;