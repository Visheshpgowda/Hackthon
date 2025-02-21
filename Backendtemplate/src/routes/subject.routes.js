import express from 'express';
import { addSubject } from '../controllers/subjectcontrollers/subject.controller.js';
import { getSubjects } from '../controllers/subjectcontrollers/subject.controller.js';

const router = express.Router();
router.post('/add', addSubject);
router.get('/get', getSubjects);
export default router;
